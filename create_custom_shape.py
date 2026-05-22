import bpy
import os
import numpy as np
import math

base_directory = os.path.dirname(os.path.abspath(__file__))
output_directory = os.path.join(base_directory, "paired_images")

def create_object():
    # Cube
    base_size = 1.65
    bpy.ops.mesh.primitive_cube_add(
        size=1,
        location=(-0.5, -0.7, 0),
        rotation=(0, 0, math.pi / 4),
        scale=(base_size, base_size, base_size)
    )

    # Store a reference to the created object
    sampled_object = bpy.context.active_object

    # Create new material
    material = bpy.data.materials.new(name="MyMaterial")

    # Access Principled BSDF shader
    bsdf = material.node_tree.nodes["Principled BSDF"]

    # Set properties of the object
    color = np.random.uniform(0.15,0.6)
    bsdf.inputs["Base Color"].default_value = (color,color,color,1)
    bsdf.inputs["Metallic"].default_value = np.random.uniform(0.0,1.0)

    # Assign material to object
    sampled_object.data.materials.append(material)

    return sampled_object

def create_camera_light(used_object):
    # We create a random light source
    light_data = bpy.data.lights.new(name="Lamp", type='POINT')
    light_object = bpy.data.objects.new(name="Lamp", object_data=light_data)
    bpy.context.collection.objects.link(light_object)

    # Set brightness of the light
    light_data.energy = 5000

    camera_data = bpy.data.cameras.new(name="Camera")
    camera_object = bpy.data.objects.new(name="Camera", object_data=camera_data)
    bpy.context.collection.objects.link(camera_object)

    # This script aligns the camera to the object.
    constraint = camera_object.constraints.new(type='TRACK_TO')
    constraint.target = used_object

    # Place the light source and the camera around the object
    light_object.location = (np.random.uniform(2.5, 5),
                             np.random.uniform(2.5, 5),
                             np.random.uniform(1.0, 5.0)
                             )
    camera_object.location = (3.0,
                              3.0,
                              3.0
                             )

    return camera_object

# By default, blender already contains a cube, camera and light in a scene.
# Therefore, we must delete all existing objects first.
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
bpy.context.scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0, 0, 0, 0)

# We set the rendering resolution to the resolution of the scraped image
r = bpy.context.scene.render
r.resolution_x = 257
r.resolution_y = 196

# Generate an object and a camera viewpoint.
current_object = create_object()
generated_camera = create_camera_light(current_object)

# Render the scene using the generated camera point
bpy.context.scene.camera = generated_camera
bpy.context.scene.render.filepath = f"{base_directory}/paired_images/custom_shape.png"
bpy.ops.render.render(write_still=True)