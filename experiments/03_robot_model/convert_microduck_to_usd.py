import math
from pathlib import Path
import mujoco
from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt

# Paths
SOURCE_XML = Path("/home/shubham/robotics/microduck-project/microduck_rl/src/mjlab_microduck/robot/microduck/robot_walk.xml")
OUTPUT_DIR = Path(__file__).resolve().parent / "assets"
OUTPUT_USD = OUTPUT_DIR / "microduck.usd"
OUTPUT_USDA = OUTPUT_DIR / "microduck.usda"


def convert_microduck():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(SOURCE_XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # 1. Create root stage
    if OUTPUT_USD.exists():
        OUTPUT_USD.unlink()
    stage = Usd.Stage.CreateNew(str(OUTPUT_USD))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # 2. Articulation Root
    robot_path = "/Robot"
    robot_prim = UsdGeom.Xform.Define(stage, robot_path)
    stage.SetDefaultPrim(robot_prim.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(robot_prim.GetPrim())

    joints_scope_path = f"{robot_path}/joints"
    stage.DefinePrim(joints_scope_path, "Scope")

    # Reference pose of trunk_base (body 1) to make /Robot root frame origin-centered
    trunk_world_pos = data.xpos[1].copy()

    # 3. Rigid Bodies (15 bodies with exact assembled poses relative to Robot root)
    body_paths = {}
    for i in range(1, model.nbody):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        bpath = f"{robot_path}/{bname}"
        body_paths[bname] = bpath

        bxform = UsdGeom.Xform.Define(stage, bpath)
        bprim = bxform.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(bprim)

        mass_api = UsdPhysics.MassAPI.Apply(bprim)
        mass_api.CreateMassAttr(float(model.body_mass[i]))
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in model.body_ipos[i]]))
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in model.body_inertia[i][:3]]))

        # Assembled pose relative to trunk_base origin (trunk_base starts at 0, 0, 0)
        rel_pos = data.xpos[i] - trunk_world_pos
        quat = data.xquat[i]
        bxform.AddTranslateOp().Set(Gf.Vec3d(float(rel_pos[0]), float(rel_pos[1]), float(rel_pos[2])))
        bxform.AddOrientOp().Set(Gf.Quatf(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])))

        # -------------------------------------------------------------
        # Solid Body Colliders (Torso, Head, Legs, and Feet)
        # -------------------------------------------------------------
        coll_specs = {
            "trunk_base": ((-0.0062, -0.0003, -0.0010), (1.0, 0.0, 0.0, 0.0), (0.0813, 0.0640, 0.0861)),
            "jaw_soft": ((0.0122, 0.0, -0.0228), (1.0, 0.0, 0.0, 0.0), (0.0593, 0.0918, 0.1227)),
            "leg": ((0.0, 0.0208, -0.0100), (1.0, 0.0, 0.0, 0.0), (0.0200, 0.0615, 0.0320)),
            "leg_2": ((-0.0208, 0.0, -0.0100), (1.0, 0.0, 0.0, 0.0), (0.0615, 0.0200, 0.0320)),
            "upper_leg_left": ((0.0104, 0.0183, 0.0120), (1.0, 0.0, 0.0, 0.0), (0.0477, 0.0610, 0.0320)),
            "upper_leg_right": ((-0.0104, 0.0183, 0.0120), (1.0, 0.0, 0.0, 0.0), (0.0477, 0.0610, 0.0320)),
            # Foot sole rectangular plates matching CAD sole_left / sole_right
            "ankle_left": ((-0.006832, -0.01821844, -0.01574058), (0.49239761, -0.50811044, -0.48982415, 0.50935319), (0.01315, 0.04124, 0.05411)),
            "ankle_right": ((0.00683211, -0.01821845, -0.01574067), (0.50935346, 0.48982405, 0.50811157, 0.49239626), (0.01315, 0.04124, 0.05411)),
        }
        if bname in coll_specs:
            c_pos, c_rot, c_size = coll_specs[bname]
            coll = UsdGeom.Cube.Define(stage, f"{bpath}/collision_box")
            coll.CreateSizeAttr(1.0)
            coll.AddTranslateOp().Set(Gf.Vec3d(*c_pos))
            coll.AddOrientOp().Set(Gf.Quatf(c_rot[0], c_rot[1], c_rot[2], c_rot[3]))
            coll.AddScaleOp().Set(Gf.Vec3f(*c_size))
            UsdPhysics.CollisionAPI.Apply(coll.GetPrim())
            UsdGeom.Imageable(coll.GetPrim()).CreateVisibilityAttr(UsdGeom.Tokens.invisible)

    # 4. Visual Meshes (73 CAD visual meshes directly from MuJoCo mesh_vert and mesh_face)
    for g_i in range(model.ngeom):
        if model.geom_type[g_i] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g_i)
        if gname in ["left_foot_collision", "right_foot_collision"]:
            continue

        m_id = model.geom_dataid[g_i]
        adr = model.mesh_vertadr[m_id]
        num = model.mesh_vertnum[m_id]
        verts = model.mesh_vert[adr : adr + num]

        fadr = model.mesh_faceadr[m_id]
        fnum = model.mesh_facenum[m_id]
        faces = model.mesh_face[fadr : fadr + fnum]

        b_id = model.geom_bodyid[g_i]
        b_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b_id)

        mesh_prim = UsdGeom.Mesh.Define(stage, f"{robot_path}/{b_name}/visual_{g_i}")
        mesh_prim.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(verts.astype("float32")))
        mesh_prim.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.flatten().astype("int32")))
        mesh_prim.CreateFaceVertexCountsAttr(Vt.IntArray([3] * fnum))

        # Color from MuJoCo material
        mat_id = model.geom_matid[g_i]
        if mat_id >= 0:
            rgba = model.mat_rgba[mat_id]
            mesh_prim.CreateDisplayColorAttr([Gf.Vec3f(float(rgba[0]), float(rgba[1]), float(rgba[2]))])
        else:
            mesh_prim.CreateDisplayColorAttr([Gf.Vec3f(0.8, 0.8, 0.8)])

        # Geom offset relative to body
        pos = model.geom_pos[g_i]
        quat = model.geom_quat[g_i]
        xform = UsdGeom.Xformable(mesh_prim)
        xform.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
        xform.AddOrientOp().Set(Gf.Quatf(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])))

    # 5. Author all 14 Revolute Joints inside /Robot/joints
    for i in range(model.njnt):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
            continue

        body_id = model.jnt_bodyid[i]
        child_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        parent_id = model.body_parentid[body_id]
        parent_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id)

        jpath = f"{joints_scope_path}/{jname}"
        revolute = UsdPhysics.RevoluteJoint.Define(stage, jpath)
        revolute.CreateAxisAttr("Z")
        revolute.CreateBody0Rel().SetTargets([body_paths[parent_name]])
        revolute.CreateBody1Rel().SetTargets([body_paths[child_name]])

        # Limits in degrees
        lower_deg = math.degrees(model.jnt_range[i][0])
        upper_deg = math.degrees(model.jnt_range[i][1])
        revolute.CreateLowerLimitAttr(lower_deg)
        revolute.CreateUpperLimitAttr(upper_deg)

        # Attachment transforms in joint frames
        pos = model.body_pos[body_id]
        quat = model.body_quat[body_id]
        revolute.CreateLocalPos0Attr(Gf.Vec3f(float(pos[0]), float(pos[1]), float(pos[2])))
        revolute.CreateLocalRot0Attr(Gf.Quatf(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])))
        revolute.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        revolute.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    stage.GetRootLayer().Save()
    stage.Export(str(OUTPUT_USDA))
    print(f"Successfully generated full Microduck USD with 15 bodies, 73 visual meshes, 8 solid colliders, and 14 joints at {OUTPUT_USD} (and exported {OUTPUT_USDA})")


if __name__ == "__main__":
    convert_microduck()
