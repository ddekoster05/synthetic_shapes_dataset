import bpy
import os
import numpy as np
import math

np.random.seed(42)

base_directory = os.path.dirname(os.path.abspath(__file__))
output_directory = os.path.join(base_directory, "samples")

classes = ['cube', 'pyramid', 'cylinder', 'cone', 'sphere', 'ring']
sample_amount = 1000


def create_object(object_type):
    match object_type:
        case 'cube':
            # Cube
            base_size = np.random.uniform(0.3,1)
            bpy.ops.mesh.primitive_cube_add(
                size=1,
                location=(0, 0, 0),
                scale=(base_size, base_size, base_size)
            )
        case 'pyramid':
            # Pyramid
            # Since cube and pyramid are 'linked classes', the sampling sizes and parameters should be identical,
            # otherwise model shortcuts may occur.
            base_size = np.random.uniform(0.3,1)
            radius = base_size / np.sqrt(2)
            bpy.ops.mesh.primitive_cone_add(
                vertices=4,
                radius1=radius,
                depth=1,
                location=(0, 0, 0),
                scale=(1,1,np.random.uniform(0.3,3))
            )
        case 'cylinder':
            # Cylinder
            # Again, cylinder and cone are linked classes, thus their sampling parameters are identical.
            bpy.ops.mesh.primitive_cylinder_add(
                vertices=64,
                radius=np.random.uniform(0.3,0.5),
                depth=np.random.uniform(0.3,3),
                location=(0, 0, 0)
            )
        case 'cone':
            # Cone
            bpy.ops.mesh.primitive_cone_add(
                vertices=64,
                radius1=np.random.uniform(0.3,0.5),
                radius2=0,
                depth=np.random.uniform(0.3,3),
                location=(0, 0, 0)
            )
        case 'sphere':
            # Sphere
            bpy.ops.mesh.primitive_uv_sphere_add(
                radius=np.random.uniform(0.3,2),
                location=(0, 0, 0),
            )
        case 'ring':
            # Ring
            bpy.ops.mesh.primitive_torus_add(
                major_radius=(np.random.uniform(1.2,2)),
                minor_radius=np.random.uniform(0.3, 1),
                location=(0, 0, 0)
            )

    # Store a reference to the created object
    sampled_object = bpy.context.active_object

    # Create new material
    material = bpy.data.materials.new(name="MyMaterial")

    # Access Principled BSDF shader
    bsdf = material.node_tree.nodes["Principled BSDF"]

    # Set properties of the object
    # We sample from a grayscale, as the pictures of real objects will also be converted to a grayscale.
    color = np.random.uniform(0.15,0.6)
    bsdf.inputs["Base Color"].default_value = (color,color,color,1)
    bsdf.inputs["Metallic"].default_value = np.random.uniform(0.0,1.0)

    # Assign material to object
    sampled_object.data.materials.append(material)

    return sampled_object

def create_camera_light(used_object, informative):
    # We create a random light source
    light_data = bpy.data.lights.new(name="Lamp", type='POINT')
    light_object = bpy.data.objects.new(name="Lamp", object_data=light_data)
    bpy.context.collection.objects.link(light_object)

    # Set brightness of the light
    light_data.energy = np.random.uniform(1000,10000)

    camera_data = bpy.data.cameras.new(name="Camera")
    camera_object = bpy.data.objects.new(name="Camera", object_data=camera_data)
    bpy.context.collection.objects.link(camera_object)

    # This script aligns the camera to the object.
    constraint = camera_object.constraints.new(type='TRACK_TO')
    constraint.target = used_object

    if informative:
        # Place the light source and the camera around the object
        light_object.location = (np.random.uniform(2.5, 5),
                                 np.random.uniform(2.5, 5),
                                 np.random.uniform(1.0, 5.0)
                                 )
        camera_object.location = (np.random.uniform(2,10),
                                  np.random.uniform(2,10),
                                  np.random.uniform(0,10)
                                  )

    else:
        # Place the light source and the camera under the object
        light_object.location = (np.random.uniform(2.5, 5),
                                 np.random.uniform(2.5, 5),
                                 np.random.uniform(-2, -5)
                                 )
        camera_object.location = (np.random.uniform(0, 0.5),
                                  np.random.uniform(0, 0.5),
                                  np.random.uniform(-2.5,-3))

        # Make the camera face upwards
        constraint.track_axis = 'TRACK_NEGATIVE_Z'
        constraint.up_axis = 'UP_Y'


    return camera_object

def generate_samples(class_name, file_location, amount, informative):
    for i in range(amount):
        # By default, blender already contains a cube, camera and light in a scene.
        # Therefore, we must delete all existing objects first.
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        bpy.context.scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0, 0, 0, 0)

        # Generate an object and a camera viewpoint.
        current_object = create_object(class_name)
        generated_camera = create_camera_light(current_object, informative)

        # Generate a string whether a view is informative, such that this can be used for identifying samples
        if informative:
            informative_string = 'informative'
        else:
            informative_string= 'uninformative'

        # Render the scene using the generated camera point
        bpy.context.scene.camera = generated_camera
        bpy.context.scene.render.filepath = f"{file_location}/{class_name}_{informative_string}_{i}.png"
        bpy.ops.render.render(write_still=True)

def build_dataset():
    for class_name in classes:
        class_directory = os.path.join(output_directory, class_name)
        if class_name == 'sphere' or class_name == 'ring':
            generate_samples(class_name, class_directory, sample_amount, True)
        else:
            informative_directory = os.path.join(class_directory, 'informative')
            generate_samples(class_name, informative_directory, sample_amount//2, True)
            uninformative_directory = os.path.join(class_directory, 'uninformative')
            generate_samples(class_name, uninformative_directory, sample_amount//2, False)

build_dataset()