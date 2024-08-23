import numpy as np
import scipy.ndimage as ndi


def load_csv(filename):
    data = np.genfromtxt(filename, delimiter=',')
    return data

# Feed in center coordinates, roll, pitch, 2theta angle, distance from sample to detector
def calc_ellipse_coefficients(x0, y0, alpha, beta, theta, distance,
                              convert=True):
    """
    All in pixel unit
    Angle in Degree
    """
    if convert:
        # Convert angles from degrees to radians
        alpha = np.deg2rad(alpha)
        beta = np.deg2rad(beta)
        theta = np.deg2rad(theta)
    # Plane vector - of the detector? Defined with rotation angles alpha, beta
    # alpha should be rotation of x if z axis is distance pointing from detector to sample via right hand rule
    a = -np.sin(alpha) * np.sin(beta)
    b = np.cos(beta) * np.sin(alpha)
    c = np.cos(alpha)
    theta2 = np.tan(2 * theta) ** 2
    # Define the components of the ellipse function - dependent on the plane vector that sliced into conic section (x-ray)
    a0 = 1 - (a ** 2 * theta2) / c ** 2
    b0 = - (2 * a * b * theta2) / c ** 2
    c0 = 1 - (b ** 2 * theta2) / c ** 2
    d0 = ((2 * a * distance * theta2) / c +
          (2 * a ** 2 * x0 * theta2) / c ** 2 +
          2 * a * b * y0 * theta2 / c ** 2 - 2 * x0)
    e0 = ((2 * b * distance * theta2) / c +
          (2 * b ** 2 * y0 * theta2) / c ** 2 +
          2 * a * b * x0 * theta2 / c ** 2 - 2 * y0)
    f0 = (x0 ** 2 + y0 ** 2 - distance ** 2 * theta2 -
          (2 * a * distance * x0 * theta2) / c -
          (a ** 2 * x0 ** 2 * theta2) / c ** 2 -
          (2 * b * distance * y0 * theta2) / c -
          (2 * a * b * x0 * y0 * theta2) / c ** 2 -
          (b ** 2 * y0 ** 2 * theta2) / c ** 2)
    return a0, b0, c0, d0, e0, f0


def get_ellipse_parameters(coefs):
    a0, b0, c0, d0, e0, f0 = coefs
    denom = b0 ** 2 - 4 * a0 * c0
    #Determinant of rotated conic tells us if parabola, hyperbola, or circle
    if denom == 0:
        return None  # Degenerate case where equals parabola; >0 then hyperbola, <0 either circle or ellipse
    # Center of the ellipse
    xc = (2 * c0 * d0 - b0 * e0) / denom
    yc = (2 * a0 * e0 - b0 * d0) / denom
    # Calculate roll angle in degrees
    roll_angle = np.degrees(np.pi / 2 +
        np.arctan2(c0 - a0 - np.sqrt((a0 - c0) ** 2 + b0 ** 2), b0))
    # Calculate lengths of major and minor axes
    term = 2 * (a0 * e0 ** 2 + c0 * d0 ** 2 - b0 * d0 * e0 + (
                b0 ** 2 - 4 * a0 * c0) * f0)
    a_major = -np.sqrt(
        term * (a0 + c0 + np.sqrt((a0 - c0) ** 2 + b0 ** 2))) / denom
    b_minor = -np.sqrt(
        term * (a0 + c0 - np.sqrt((a0 - c0) ** 2 + b0 ** 2))) / denom
    return a_major, b_minor, xc, yc, roll_angle


def energy_kev_to_theta(energy_kev, d_spacing_angstrom, convert=True):
    """
    output is degree

    """
    h = 6.62607015e-34
    c = 299792458
    energy_j = energy_kev * 1000 * 1.60218e-19
    wave_length = h * c / energy_j
    theta = np.arcsin(wave_length / (2 * d_spacing_angstrom * 1e-10))
    if convert:
        theta = np.rad2deg(theta)
    return theta


def generate_ellipse_image(coefs, height, width, tolerance=0.1, size=3,
                           max_val=1.0):
    x_list = 1.0 * np.arange(width)
    y_list = 1.0 * np.arange(height)[::-1]
    x_mat, y_mat = np.meshgrid(x_list, y_list)
    a0, b0, c0, d0, e0, f0 = coefs
    image = a0 * x_mat ** 2 + b0 * x_mat * y_mat + c0 * y_mat ** 2 + \
            d0 * x_mat + e0 * y_mat + f0
    image = ndi.gaussian_filter(np.clip(image, -tolerance, tolerance), 2)
    image = np.float32(ndi.gaussian_gradient_magnitude(image, size))
    num = np.max(image)
    if num != 0.0:
        image = image / np.max(image)
    return np.float32(image * max_val)
