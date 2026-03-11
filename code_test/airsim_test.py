import airsim
import time
import math

client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True, "Car1")

def get_yaw():
    state = client.getCarState("Car1")
    ori = state.kinematics_estimated.orientation
    x, y, z, w = ori.x_val, ori.y_val, ori.z_val, ori.w_val
    yaw = math.atan2(2.0 * (w*z + x*y), 1.0 - 2.0*(y*y + z*z))
    return math.degrees(yaw)

print("测试1：获取初始Yaw")
print(f"初始Yaw: {get_yaw():.1f}° (应该接近0°)")

print("\n测试2：转向控制测试")
print("steering=1.0 (正值)...")

print("\n测试2B：先拉手刹，再steering=±1")
controls = airsim.CarControls()
controls.throttle = 0.0
controls.steering = 0.0
controls.handbrake = True
client.setCarControls(controls)

client.setCarControls(airsim.CarControls(throttle=0.0, steering=1.0))
time.sleep(2.0)
yaw1 = get_yaw()
print(f"Yaw变化: {yaw1:.1f}°")

print("steering=-1.0 (负值)...")
client.setCarControls(airsim.CarControls(throttle=0.0, steering=-1.0))
time.sleep(2.0)
yaw2 = get_yaw()
print(f"Yaw变化: {yaw2:.1f}°")

client.enableApiControl(False)