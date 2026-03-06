import bpy
import os

base_directory = os.path.dirname(os.path.abspath(__file__))
output_directory = os.path.join(base_directory, "samples")

classes = ['cube', 'pyramid', 'cylinder', 'cone', 'sphere', 'ring']
sample_amount = 1

# TO_DO: ADD PARAMETERS TO SAMPLE FROM!

def create_object(object_type):
    # TO_DO: MAKE PARAMETERS ABLE TO BE SAMPLED
    match object_type:
        case 'cube':
            # Cube
            bpy.ops.mesh.primitive_cube_add(
                size=2.0,
                location=(0, 0, 0)
            )
        case 'pyramid':
            bpy.ops.mesh.primitive_cone_add(
                vertices =4,
                radius1=1,
                depth=2,
                location=(0, 0, 0),
                scale=(1,1,0.8)
            )
        case 'cylinder':
            # Cylinder
            bpy.ops.mesh.primitive_cylinder_add(
                # Vertices should stay constant!!!
                vertices=32,
                radius=1,
                depth=2,
                location=(0, 0, 0)
            )
        case 'cone':
            # Cone
            bpy.ops.mesh.primitive_cone_add(
                vertices=32,
                radius1=1,
                radius2=0,
                depth=2,
                location=(0, 0, 0)
            )
        case 'sphere':
            # Sphere
            bpy.ops.mesh.primitive_uv_sphere_add(
                radius=1,
                location=(0, 0, 0),
            )
        case 'ring':
            # Ring
            bpy.ops.mesh.primitive_torus_add(
                major_radius=1,
                minor_radius=0.3,
                location=(0, 0, 0)
            )

    # Store a reference to the created object
    sampled_object = bpy.context.active_object

    # TO_DO: Make material properties able to be sampled
    # Create new material
    material = bpy.data.materials.new(name="MyMaterial")

    # Access Principled BSDF shader
    bsdf = material.node_tree.nodes["Principled BSDF"]

    # Set properties of the object
    bsdf.inputs["Base Color"].default_value = (1, 0, 0, 1)
    bsdf.inputs["Metallic"].default_value = 0.3

    # Assign material to object
    sampled_object.data.materials.append(material)

    return sampled_object

def create_camera_light(used_object):
    # TO_DO: MAKE LIGHT PARAMETERS ABLE TO BE SAMPLED
    # We create a random light source
    light_data = bpy.data.lights.new(name="Lamp", type='POINT')
    light_object = bpy.data.objects.new(name="Lamp", object_data=light_data)
    bpy.context.collection.objects.link(light_object)

    # Set properties of the light
    light_object.location = (0, 0, 5)
    light_data.energy = 1000

    # TO_DO: ADD SAMPLE RANGES FOR BOTH AMBIGUOUS AND UNAMBIGUOUS VIEWS
    camera_data = bpy.data.cameras.new(name="Camera")
    camera_object = bpy.data.objects.new(name="Camera", object_data=camera_data)
    bpy.context.collection.objects.link(camera_object)

    camera_object.location = (5, 5, 5)

    # This script aligns the camera to the object.
    constraint = camera_object.constraints.new(type='TRACK_TO')
    constraint.target = used_object

    return camera_object


for class_name in classes:
    # TO_DO: AUTOMATE CREATING SAMPLES FOR AMBIGUOUS AND UNAMBIGUOUS SHAPES

    class_directory = os.path.join(output_directory, class_name)

    for i in range(sample_amount):
        # By default, blender already contains a cube, camera and light in a scene.
        # Therefore, we must delete all existing objects first.
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)

        # Generate an object and a camera viewpoint.
        current_object = create_object(class_name)
        generated_camera = create_camera_light(current_object)

        # Render the scene using the generated camera point
        bpy.context.scene.camera = generated_camera
        bpy.context.scene.render.filepath = f"{class_directory}/{class_name}_{i}.png"
        bpy.ops.render.render(write_still=True)