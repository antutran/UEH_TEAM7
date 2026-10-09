#!/usr/bin/env python3
"""Map the Yahboom firmware speed loop (set_car_motion, car type X3) onto this car's motors.
Wheels must be lifted. For each command, prints the measured rate of all four motor ports
(rad/s at 1320 counts/rev), the board's own motion estimate, and the step response of M1/M4."""
import math
import statistics as st
import time

from crc_bringup.Rosmaster_Lib import Rosmaster

CPR = 1320.0
CMDS = [(0.2, 0.0), (-0.2, 0.0), (0.0, 1.0), (0.0, -1.0)]
HOLD = 2.5

bot = Rosmaster(car_type=1, com='/dev/myserial')
bot.create_receive_threading()
time.sleep(0.5)
print('car type %s, battery %.1f V' % (bot.get_car_type_from_machine(), bot.get_battery_voltage()), flush=True)


def rates(samples):
    (t0, e0), (t1, e1) = samples[0], samples[-1]
    return [(e1[i] - e0[i]) * 2 * math.pi / CPR / (t1 - t0) for i in range(4)]


try:
    for vx, vz in CMDS:
        log = []
        t0 = time.monotonic()
        while time.monotonic() - t0 < HOLD:
            bot.set_car_motion(vx, 0.0, vz)
            time.sleep(0.04)
            log.append((time.monotonic() - t0, bot.get_motor_encoder()))
        steady = [s for s in log if s[0] > 1.0]
        r = rates(steady)
        print('cmd vx=%+.2f vz=%+.2f | steady rad/s M1 %+6.2f M2 %+6.2f M3 %+6.2f M4 %+6.2f | board motion %s' % (
            vx, vz, r[0], r[1], r[2], r[3], [round(x, 3) for x in bot.get_motion_data()]), flush=True)
        # step response of M1 and M4 from 0.12 s windows
        trace = []
        for i in range(3, len(log)):
            (ta, ea), (tb, eb) = log[i - 3], log[i]
            trace.append((tb, [(eb[k] - ea[k]) * 2 * math.pi / CPR / (tb - ta) for k in (0, 3)]))
        for k, name in ((0, 'M1'), (1, 'M4')):
            target = r[0] if k == 0 else r[3]
            if abs(target) < 0.5:
                continue
            ys = [(t, v[k] / target) for t, v in trace]
            peak = max(y for _, y in ys)
            settled = next((t for t, _ in ys if all(abs(y2 - 1) < 0.15 for t2, y2 in ys if t2 >= t)), None)
            tail = [y for t, y in ys if t > 1.0]
            print('    %s: overshoot %+3.0f %%, settled %s, ripple %.0f %%' % (
                name, 100 * (peak - 1), '%.2f s' % settled if settled else 'never', 100 * st.pstdev(tail)), flush=True)
        bot.set_car_motion(0, 0, 0)
        time.sleep(1.0)
finally:
    bot.set_car_motion(0, 0, 0)
    bot.set_motor(0, 0, 0, 0)
    time.sleep(0.2)
    print('stopped', flush=True)
