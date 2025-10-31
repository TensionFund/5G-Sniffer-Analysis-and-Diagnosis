#!/usr/bin/env python3
# check_usrp_sensors_verbose.py
# Robustly prints USRP master clock, time/clock sources, and sensors across UHD/python versions.

import sys
try:
    import uhd
except Exception as e:
    print("ERROR: cannot import uhd (pyuhd). Is it installed? Exception:", e)
    sys.exit(1)

def pp_sensor(sname, sensor):
    print("  Sensor name:", sname)
    try:
        # Try common methods across versions
        if hasattr(sensor, "to_pp_string"):
            print("    to_pp_string():", sensor.to_pp_string())
        if hasattr(sensor, "to_string"):
            print("    to_string():", sensor.to_string())
        # Some versions have .to_pp or .value
        if hasattr(sensor, "value"):
            try:
                print("    value:", sensor.value)
            except Exception:
                pass
        # fallback to str()
        print("    str():", str(sensor))
        # try dictionary-style attributes
        if hasattr(sensor, "__dict__"):
            print("    __dict__:", sensor.__dict__)
    except Exception as ex:
        print("    [error printing sensor]:", ex)

def main():
    args = 'type=b200,clock=external,sync=external,master_clock_rate=23040000'
    if len(sys.argv) > 1:
        args = sys.argv[1]
    print("Opening USRP with args:", args)
    usrp = uhd.usrp.MultiUSRP(args)
    # Print master clock rate
    try:
        m = usrp.get_master_clock_rate()
        print("Master clock rate (Hz):", m)
    except Exception as e:
        print("Cannot get_master_clock_rate():", e)

    # get_time_sources signature differences: some require channel index
    try:
        ts = usrp.get_time_sources(0)
    except TypeError:
        try:
            ts = usrp.get_time_sources()
        except Exception as e:
            ts = None
            print("get_time_sources failed:", e)
    print("Time sources (channel 0):", ts)

    try:
        cs = usrp.get_clock_sources(0)
    except TypeError:
        try:
            cs = usrp.get_clock_sources()
        except Exception as e:
            cs = None
            print("get_clock_sources failed:", e)
    print("Clock sources (channel 0):", cs)

    print("\n=== MBoard sensors ===")
    mboard_names = []
    try:
        # get_mboard_sensor_names exists in some versions
        if hasattr(usrp, "get_mboard_sensor_names"):
            mboard_names = usrp.get_mboard_sensor_names(0)
        else:
            # fallback attempt
            mboard_names = ["ref_locked", "lo_locked"]
    except Exception:
        mboard_names = ["ref_locked", "lo_locked"]
    for name in mboard_names:
        try:
            s = usrp.get_mboard_sensor(name, 0)
            pp_sensor(name, s)
        except Exception as e:
            print("  Unable to read mboard sensor", name, ":", e)

    print("\n=== LO / frontend sensors (try common names) ===")
    common = ["lo_locked", "lo_locked_ch0", "lo_locked_ch1", "ref_locked"]
    for name in common:
        try:
            s = usrp.get_sensor(name, 0)
            pp_sensor(name, s)
        except Exception:
            try:
                s = usrp.get_mboard_sensor(name, 0)
                pp_sensor(name, s)
            except Exception as e:
                print("  no sensor", name, ":", e)

    print("\n=== Done ===\n")
    print("Notes:")
    print(" - If any 'ref_locked' or 'lo_locked' sensor prints True / '1' / 'locked' then device is locked to external 10MHz.")
    print(" - If the sensor values are ambiguous, paste this output and I will interpret.")

if __name__ == "__main__":
    main()

