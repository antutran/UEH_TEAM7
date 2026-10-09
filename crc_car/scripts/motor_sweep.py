#!/usr/bin/env python3
"""Open-loop motor characterisation for the Yahboom ROS board: for each motor and PWM step,
measure the raw encoder rate. Wheels must be lifted. Raw values: no inversion applied."""
import math
import time

from crc_bringup.Rosmaster_Lib import Rosmaster

COUNTS_PER_REV = 1320.0
MOTORS = (1, 4)
STEPS = (5, 8, 10, 15, 20, 30, 50, 70, -5, -8, -10, -15, -20, -30, -50, -70)
SETTLE, MEASURE = 1.0, 1.0

bot = Rosmaster(car_type=1, com="/dev/myserial")
bot.create_receive_threading()
time.sleep(0.5)
print("battery %.1f V" % bot.get_battery_voltage(), flush=True)


def drive(motor, pwm):
    cmd = [0, 0, 0, 0]
    cmd[motor - 1] = pwm
    bot.set_motor(*cmd)


try:
    for motor in MOTORS:
        print("motor M%d" % motor, flush=True)
        for pwm in STEPS:
            drive(motor, pwm)
            time.sleep(SETTLE)
            c0, t0 = bot.get_motor_encoder()[motor - 1], time.monotonic()
            time.sleep(MEASURE)
            c1, t1 = bot.get_motor_encoder()[motor - 1], time.monotonic()
            rate = (c1 - c0) / (t1 - t0)
            print("  pwm %+4d %%  counts/s %+7.0f  wheel %+6.2f rad/s" % (
                pwm, rate, rate * 2 * math.pi / COUNTS_PER_REV), flush=True)
            drive(motor, 0)
            time.sleep(0.4)
finally:
    bot.set_motor(0, 0, 0, 0)
    time.sleep(0.2)
    bot.set_motor(0, 0, 0, 0)
    print("motors stopped", flush=True)
