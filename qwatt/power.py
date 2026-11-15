"""Power sampling: Intel RAPL through Windows 'Energy Meter' PDH counters (no admin).

Verified units (scripts/smoke_power.py): `Power` is milliwatts; `Energy` is picowatt-hours
(1 raw unit = 1e-12 Wh = 3.6e-9 J). We store pkg_joules_cum in joules.

The sampler runs as a SEPARATE process (scripts/run_sampler.py -> run_sampler()) and writes to SQLite.
`ensure_sampler()` spawns it detached if no heartbeat in the last 2 s; safe to call repeatedly.
Tier 'modeled' (CPU-utilisation model) is used only if PDH is unavailable and is labeled as such.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from qwatt import config, store

J_PER_RAW = 3.6e-9           # picowatt-hour -> joule
MW_TO_W = 1e-3
PDH_INSTANCES = {"pkg": "RAPL_Package0_PKG", "pp0": "RAPL_Package0_PP0", "pp1": "RAPL_Package0_PP1"}

SPAWN_MARK = config.ROOT / "logs" / ".sampler_spawn"
HEARTBEAT_MAX_AGE = 2.0


class PdhRapl:
    """Reads RAPL package/core/gpu power (W) and cumulative package energy (J) via win32pdh."""

    def __init__(self):
        import win32pdh
        self._w = win32pdh
        self.q = win32pdh.OpenQuery()
        self.h: dict[tuple[str, str], int] = {}
        for k, inst in PDH_INSTANCES.items():
            for c in ("Power", "Energy"):
                try:
                    self.h[(k, c)] = win32pdh.AddCounter(self.q, f"\\Energy Meter({inst})\\{c}")
                except Exception:
                    if k == "pkg":
                        raise
        win32pdh.CollectQueryData(self.q)
        self.source = "rapl_pdh"

    def read(self) -> dict:
        w = self._w
        w.CollectQueryData(self.q)
        v = {}
        for key, h in self.h.items():
            try:
                v[key] = w.GetFormattedCounterValue(h, w.PDH_FMT_DOUBLE)[1]
            except Exception:
                v[key] = None
        e = v.get(("pkg", "Energy"))
        if e is None:
            raise RuntimeError("PDH energy counter unreadable")
        return {
            "pkg_watts": (v.get(("pkg", "Power")) or 0.0) * MW_TO_W,
            "pp0_watts": (v.get(("pp0", "Power")) or 0.0) * MW_TO_W if v.get(("pp0", "Power")) is not None else None,
            "pp1_watts": (v.get(("pp1", "Power")) or 0.0) * MW_TO_W if v.get(("pp1", "Power")) is not None else None,
            "pkg_joules_cum": e * J_PER_RAW,   # absolute joules since boot (monotonic across sampler restarts)
            "source": self.source,
        }

    def close(self):
        try:
            self._w.CloseQuery(self.q)
        except Exception:
            pass


class ModeledPower:
    """LAST RESORT tier: package watts modeled from CPU utilisation (labeled source='modeled')."""

    IDLE_W, MAX_W = 5.0, 30.0

    class _FT(ctypes.Structure):
        _fields_ = [("lo", ctypes.c_uint32), ("hi", ctypes.c_uint32)]

    def __init__(self):
        self.source = "modeled"
        self._last = self._times()
        self._cum = 0.0
        self._t = time.time()

    def _times(self):
        i, k, u = self._FT(), self._FT(), self._FT()
        ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(i), ctypes.byref(k), ctypes.byref(u))
        f = lambda x: (x.hi << 32) | x.lo
        return f(i), f(k), f(u)

    def read(self) -> dict:
        idle, kern, user = self._times()
        li, lk, lu = self._last
        self._last = (idle, kern, user)
        tot = (kern - lk) + (user - lu)
        util = 0.0 if tot <= 0 else max(0.0, min(1.0, 1.0 - (idle - li) / tot))
        w = self.IDLE_W + (self.MAX_W - self.IDLE_W) * util
        now = time.time()
        self._cum += w * (now - self._t)
        self._t = now
        return {"pkg_watts": w, "pp0_watts": None, "pp1_watts": None,
                "pkg_joules_cum": self._cum, "source": "modeled"}

    def close(self):
        pass


class BatteryReader:
    """Battery DischargeRate (mW -> W) from root\\wmi BatteryStatus, polled in a background thread
    at ~1 Hz so the 4 Hz loop never waits on WMI. value is None unless discharging."""

    def __init__(self, period: float = 1.0):
        self.value: float | None = None
        self._period = period
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def _run(self):
        try:
            import pythoncom
            import win32com.client
            pythoncom.CoInitialize()
            svc = win32com.client.GetObject("winmgmts:{impersonationLevel=impersonate}!//./root/wmi")
        except Exception:
            return
        while not self._stop.is_set():
            try:
                rows = svc.ExecQuery("SELECT DischargeRate, Discharging FROM BatteryStatus")
                val = None
                for r in rows:
                    if bool(r.Discharging) and r.DischargeRate and r.DischargeRate > 0:
                        val = float(r.DischargeRate) * 1e-3
                        break
                self.value = val
            except Exception:
                self.value = None
            self._stop.wait(self._period)

    def stop(self):
        self._stop.set()


def make_reader():
    """PDH RAPL if available, else the labeled modeled fallback."""
    try:
        r = PdhRapl()
        time.sleep(0.15)
        r.read()
        return r
    except Exception:
        return ModeledPower()


# --------------------------------------------------------------------------- sampler loop
def run_sampler(hz: float | None = None, db_path=None, max_seconds: float | None = None) -> None:
    """Sampler main loop (blocks). One instance only: exits if a live heartbeat from another pid exists."""
    hz = hz or config.Settings().sampler_hz
    period = 1.0 / hz
    con = store.connect(db_path)
    hb = store.get_heartbeat(con)
    if hb and hb["pid"] != os.getpid() and time.time() - hb["ts"] < HEARTBEAT_MAX_AGE * 2 and _pid_alive(hb["pid"]):
        return
    reader = make_reader()
    battery = BatteryReader()
    start = time.time()
    nxt = time.perf_counter()
    buf: list[dict] = []
    try:
        while True:
            try:
                r = reader.read()
            except Exception:
                if reader.source != "modeled":
                    reader = ModeledPower()   # PDH died mid-run; fall back, labeled
                    continue
                raise
            ts = time.time()
            r["ts"] = ts
            r["sys_watts"] = battery.value
            buf.append(r)
            try:
                store.insert_samples(con, buf)
                buf.clear()
                store.set_heartbeat(con, ts, os.getpid(), reader.source)
            except Exception:
                buf = buf[-200:]
            nxt += period
            d = nxt - time.perf_counter()
            if d > 0:
                time.sleep(d)
            else:
                nxt = time.perf_counter()
            if max_seconds and time.time() - start > max_seconds:
                break
    finally:
        battery.stop()
        reader.close()
        con.close()


def _pid_alive(pid: int) -> bool:
    try:
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))  # QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(h)
        return code.value == 259  # STILL_ACTIVE
    except Exception:
        return False


# --------------------------------------------------------------------------- spawn / ensure
def heartbeat_age(con=None) -> float | None:
    own = con is None
    con = con or store.connect()
    try:
        hb = store.get_heartbeat(con)
        return None if hb is None else time.time() - hb["ts"]
    finally:
        if own:
            con.close()


def sampler_source(con=None) -> str | None:
    own = con is None
    con = con or store.connect()
    try:
        hb = store.get_heartbeat(con)
        return hb["source"] if hb else None
    finally:
        if own:
            con.close()


def ensure_sampler(wait_s: float = 4.0) -> bool:
    """Start the sampler process if there is no heartbeat in the last 2 s. Safe to call repeatedly
    (a spawn marker prevents double-starting). Returns True if a fresh heartbeat exists on return."""
    def fresh() -> bool:
        try:
            a = heartbeat_age()
            return a is not None and a < HEARTBEAT_MAX_AGE
        except Exception:
            return False

    if fresh():
        return True
    try:
        SPAWN_MARK.parent.mkdir(parents=True, exist_ok=True)
        recently = SPAWN_MARK.exists() and time.time() - SPAWN_MARK.stat().st_mtime < 8.0
    except Exception:
        recently = False
    if not recently:
        try:
            SPAWN_MARK.write_text(str(time.time()))
        except Exception:
            pass
        script = config.ROOT / "scripts" / "run_sampler.py"
        flags = 0
        if sys.platform == "win32":
            flags = 0x00000008 | 0x08000000 | 0x00000200  # DETACHED | NO_WINDOW | NEW_PROCESS_GROUP
        log = open(config.ROOT / "logs" / "sampler.log", "ab") if (config.ROOT / "logs").exists() else subprocess.DEVNULL
        subprocess.Popen([sys.executable, str(script)], cwd=str(config.ROOT), creationflags=flags,
                         stdin=subprocess.DEVNULL, stdout=log, stderr=log, close_fds=True)
    end = time.time() + wait_s
    while time.time() < end:
        if fresh():
            return True
        time.sleep(0.15)
    return fresh()
