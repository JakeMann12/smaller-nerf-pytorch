import numpy as np

def matrix_to_euler_angles(matrix):
    # Extract the rotation matrix portion from the transformation matrix
    rotation_matrix = matrix[:3, :3]

    # Compute Euler angles from the rotation matrix
    sy = np.sqrt(rotation_matrix[0,0] * rotation_matrix[0,0] + rotation_matrix[1,0] * rotation_matrix[1,0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(rotation_matrix[2,1], rotation_matrix[2,2])
        y = np.arctan2(-rotation_matrix[2,0], sy)
        z = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0])
    else:
        x = np.arctan2(-rotation_matrix[1,2], rotation_matrix[1,1])
        y = np.arctan2(-rotation_matrix[2,0], sy)
        z = 0

    return np.degrees([x, y, z])

# Example usage
T = np.array([
                [
                    0.6219741106033325,
                    -0.7788047194480896,
                    0.08130991458892822,
                    0.32777073979377747
                ],
                [
                    0.7830376029014587,
                    0.6186118721961975,
                    -0.06458522379398346,
                    -0.26035135984420776
                ],
                [
                    -3.7252898543727042e-09,
                    0.10383893549442291,
                    0.9945940971374512,
                    4.009336948394775
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0
                ]
            ])

euler_angles = matrix_to_euler_angles(T)
print("Euler angles (degrees):", euler_angles)
