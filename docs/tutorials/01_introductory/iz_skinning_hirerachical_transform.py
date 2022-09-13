import numpy as np
from fury import window, utils, actor, transform
from fury.gltf import glTF
from fury.data import fetch_gltf, read_viz_gltf
from fury.lib import Transform
import copy

scene = window.Scene()

fetch_gltf('SimpleSkin', 'glTF')
filename = read_viz_gltf('CesiumMan')

gltf_obj = glTF(filename, apply_normals=False)
actors = gltf_obj.actors()
print(len(actors))

vertices = utils.vertices_from_actor(actors[0])
clone = np.copy(vertices)

timeline = gltf_obj.get_skin_timeline3()
timeline.add_actor(actors[0])

scene = window.Scene()
showm = window.ShowManager(scene, size=(900, 768), reset_camera=False,
                           order_transparent=True)
showm.initialize()

bactors = gltf_obj.get_joint_actors(length=0.2)
bverts = {}
for bone, joint_actor in bactors.items():
    bverts[bone] = utils.vertices_from_actor(joint_actor)

bvert_copy = copy.deepcopy(bverts)

scene.add(* bactors.values())
scene.add(timeline)

bones = gltf_obj.bones[0]
parent_transforms = gltf_obj.bone_tranforms


def transverse_timelines(timeline, bone_id, timestamp,
                         parent_bone_deform=np.identity(4)):
    deform = timeline.get_value('transform', timestamp)
    new_deform = np.dot(parent_bone_deform, deform)

    node = gltf_obj.gltf.nodes[bone_id]
    bverts[bone_id][:] = transform.apply_transfomation(bvert_copy[bone_id],
                                                       new_deform)
    utils.update_actor(bactors[bone_id])
    # if bone_id == 11:
    #     print(bverts[bone_id])
    if node.children:
        for c_timeline, c_bone in zip(timeline.timelines, node.children):
            transverse_timelines(c_timeline, c_bone, timestamp, new_deform)


def timer_callback(_obj, _event):
    timeline.update_animation()
    timestamp = timeline.current_timestamp

    for child in timeline.timelines:
        transverse_timelines(child, bones[0], timestamp)
    showm.render()


showm.add_timer_callback(True, 10, timer_callback)

showm.start()
