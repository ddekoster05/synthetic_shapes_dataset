import bpy
import os
import numpy as np
import math

base_directory = os.path.dirname(os.path.abspath(__file__))
output_directory = os.path.join(base_directory, "paired_images")
shape_number = 5

def create_object(base_size, location, rotation, color):
    # Cube
    bpy.ops.mesh.primitive_cube_add(
        size=1,
        location=location,
        rotation=rotation,
        scale=(base_size, base_size, base_size)
    )

    # Store a reference to the created object
    sampled_object = bpy.context.active_object

    # Create new material
    material = bpy.data.materials.new(name="MyMaterial")

    # Access Principled BSDF shader
    bsdf = material.node_tree.nodes["Principled BSDF"]

    # Set properties of the object
    bsdf.inputs["Base Color"].default_value = (color,color,color,1)
    bsdf.inputs["Metallic"].default_value = 0

    # Assign material to object
    sampled_object.data.materials.append(material)

    return sampled_object

def create_camera_light(used_object, light_location, camera_object_location, camera_lens=50):
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
    light_object.location = light_location
    camera_object.location = camera_object_location

    camera_data.lens = camera_lens

    return camera_object

# By default, blender already contains a cube, camera and light in a scene.
# Therefore, we must delete all existing objects first.
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
bpy.context.scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0, 0, 0, 0)

# We set the rendering resolution to the resolution of the scraped image, and generate an object and a camera viewpoint.
r = bpy.context.scene.render

# Rubix cube
if shape_number == 1:
    r.resolution_x = 1024
    r.resolution_y = 1024
    current_object = create_object(1.4, (-0.2, -0.2, 0), (0, 0, 0), 0.6)
    generated_camera = create_camera_light(current_object, (5.0, 1.5, 4.5), (3.4, 3.4, 3), 80)
# Ice cube
elif shape_number == 5:
    r.resolution_x = 257
    r.resolution_y = 196
    current_object = create_object(1.65, (-0.5, -0.7, 0), (0, 0, math.pi / 4 + math.radians(2)), 0.2)
    generated_camera = create_camera_light(current_object, (6, -2.5, 4), (3.0, 3.0, 3.0), 52)

# Render the scene using the generated camera point
bpy.context.scene.camera = generated_camera
bpy.context.scene.render.filepath = f"{base_directory}/paired_images/custom_shape_5.png"
bpy.ops.render.render(write_still=True)