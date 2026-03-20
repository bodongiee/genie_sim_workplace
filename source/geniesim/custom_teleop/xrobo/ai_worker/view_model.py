import mujoco
import mujoco.viewer


xml_path = '/home/kwon/Desktop/ai_worker_geniesim/ffw_sg2.xml'
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# keyframe 0으로 초기화 (initial pose 적용)
mujoco.mj_resetDataKeyframe(model, data, 0)

# ctrl도 keyframe qpos와 동일하게 설정 (포즈 유지)
# actuator 순서: left 8개 + right 8개 (waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate, left_finger, right_finger)
data.ctrl[:] = data.qpos[:model.nu]

mujoco.mj_forward(model, data)

mujoco.viewer.launch(model, data)
