import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from django.shortcuts import render
from django import forms
from .forms import Challenge1Form, Challenge2Form, Challenge3Form, Challenge4Form, Challenge5Form, Challenge6Form, Challenge7Form, Challenge8Form, Challenge9Form, Extension1Form, Extension2Form

########### - Challenge 1 - ###########

def calculate_trajectory(u, angle, g, h):
    theta = np.radians(angle)
    t_flight = (u * np.sin(theta) + np.sqrt((u * np.sin(theta))**2 + 2 * g * h)) / g
    t = np.linspace(0, t_flight, num=500)
    x = u * np.cos(theta) * t
    y = h + u * np.sin(theta) * t - 0.5 * g * t**2
    return x, y

def challenge1_view(request):
    if request.method == 'POST':
        form = Challenge1Form(request.POST)
        if form.is_valid():
            angle = form.cleaned_data['angle']
            g = form.cleaned_data['gravity']
            u = form.cleaned_data['velocity']
            h = form.cleaned_data['height']
            x, y = calculate_trajectory(u, angle, g, h)

            plt.figure(figsize=(10, 5))
            plt.plot(x, y)
            plt.xlabel('Distance (m)')
            plt.ylabel('Height (m)')
            plt.title('Projectile Motion')
            plt.xlim(left=0)
            plt.ylim(bottom=0)
            plt.grid(True)
            plt.legend()

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            buf.close()

            return render(request, 'motion/challenge1.html', {'form': form, 'image_base64': image_base64})
    else:
        form = Challenge1Form()
    return render(request, 'motion/challenge1.html', {'form': form})

########### - Challenge 2 - ###########

def total_flight_time(angle, g, u, h):
    initial_y = u * np.sin(angle)
    discriminant = (initial_y**2) + 2 * g * h
    if discriminant < 0:
        return 0
    else:
        t1 = (initial_y + np.sqrt(discriminant)) / g
        t2 = (initial_y - np.sqrt(discriminant)) / g
        return max(t1, t2)

def projectile_motion_2(angle, g, u, h):
    angle = np.radians(angle)
    T = total_flight_time(angle, g, u, h)
    initial_x = u * np.cos(angle)
    x_range = initial_x * T
    x = np.linspace(0, x_range, 500)
    y = h + (x * np.tan(angle)) - ((g * x**2) / (2 * u**2 * np.cos(angle)**2))
    return x, y

def challenge2_view(request):
    if request.method == 'POST':
        form = Challenge2Form(request.POST)
        if form.is_valid():
            angle = form.cleaned_data['angle']
            g = form.cleaned_data['g']
            u = form.cleaned_data['u']
            h = form.cleaned_data['h']
            x, y = projectile_motion_2(angle, g, u, h)

            angle = np.radians(angle)
            initial_x = u * np.cos(angle)
            initial_y = u * np.sin(angle)
            max_height = h + ((initial_y**2) / (2 * g))
            max_range = ((initial_x) * (initial_y / g))

            plt.figure(figsize=(10, 5))
            plt.title('Projectile Motion')
            plt.plot(x, y, label='Projectile Trajectory')
            plt.plot([max_range], [max_height], 'ro', label='Apogee')
            plt.xlabel('Horizontal Distance')
            plt.ylabel('Vertical Distance')
            plt.xlim(left=0)
            plt.ylim(bottom=0)
            plt.grid(True)
            plt.legend()

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            buf.close()

            return render(request, 'motion/challenge2.html', {'form': form, 'image_base64': image_base64})
    else:
        form = Challenge2Form()
    return render(request, 'motion/challenge2.html', {'form': form})

########### - Challenge 3 - ###########
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from django.shortcuts import render
from django import forms
from .forms import Challenge3Form

def projectile_motion_3(u, angle, g):
    angle = np.radians(angle)
    initial_x = u * np.cos(angle)
    initial_y = u * np.sin(angle)
    
    flight_time = (2 * initial_y) / g
    
    T = np.linspace(0, flight_time, num=500)
    
    x = initial_x * T
    y = initial_y * T - 0.5 * g * T**2
    
    return x, y

def minimum_velocity(X, Y, g):
    min_u = np.sqrt(g * (Y + np.sqrt(X**2 + Y**2)))
    return min_u

def calculate_trajectory_angles(X, Y, u, g):
    a = (g / (2 * u**2)) * (X**2)
    b = -X
    c = Y - (g * X**2) / (2 * u**2)
    
    discriminant = b**2 - 4 * a * c
    
    if discriminant < 0:
        raise ValueError("No real solution exists for the given parameters.")
    
    sqrt_disc = np.sqrt(discriminant)
    angle1 = np.degrees(np.arctan((-b + sqrt_disc) / (2 * a)))
    angle2 = np.degrees(np.arctan((-b - sqrt_disc) / (2 * a)))
    
    return angle1, angle2

def adjust_trajectory_to_target(x_trajectory, y_trajectory, target_x, target_y):
    x_intercept = x_trajectory[np.where(np.abs(x_trajectory - target_x) < 1e-2)[0]]
    if len(x_intercept) > 0:
        index = np.where(np.abs(x_trajectory - target_x) < 1e-2)[0][0]
        return x_trajectory[:index+1], y_trajectory[:index+1]
    else:
        return x_trajectory, y_trajectory

def plot_projectile_3(X, Y, g):
    min_u = minimum_velocity(X, Y, g)

    min_angle = np.degrees(np.arctan((Y + np.sqrt(X**2 + Y**2)) / X))
    x_min_u, y_min_u = projectile_motion_3(min_u, min_angle, g)

    try:
        high_angle, low_angle = calculate_trajectory_angles(X, Y, min_u, g)

        x_high, y_high = projectile_motion_3(min_u, high_angle, g)
        x_low, y_low = projectile_motion_3(min_u, low_angle, g)

        x_min_u, y_min_u = adjust_trajectory_to_target(x_min_u, y_min_u, X, Y)
        x_high, y_high = adjust_trajectory_to_target(x_high, y_high, X, Y)
        x_low, y_low = adjust_trajectory_to_target(x_low, y_low, X, Y)

        plt.figure(figsize=(10, 5))
        plt.plot(x_min_u, y_min_u, 'grey', label='Minimum Velocity')
        plt.plot(x_high, y_high, 'blue', label='High Ball')
        plt.plot(x_low, y_low, 'red', label='Low Ball')
        plt.plot(X, Y, 'yo', label='Target', markersize=10)

        plt.title('Projectile Trajectories')
        plt.xlabel('Horizontal Distance')
        plt.ylabel('Vertical Distance')
        plt.ylim(bottom=0)
        plt.xlim(left=0)
        plt.grid(True)
        plt.legend()
        plt.show()
    except ValueError as e:
        print(e)

def challenge3_view(request):
    if request.method == 'POST':
        form = Challenge3Form(request.POST)
        if form.is_valid():
            X = form.cleaned_data['x_target']
            Y = form.cleaned_data['y_target']
            g = form.cleaned_data['g']

            print(f"Inputs - X: {X}, Y: {Y}, g: {g}")

            try:
                plot_projectile_3(X, Y, g)
                print("Plotting successful")

                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                buf.close()
                
                print("Image successfully encoded and passed to template")
                return render(request, 'motion/challenge3.html', {'form': form, 'image_base64': image_base64})
            except Exception as e:
                print(f"Error during plotting: {e}")
                return render(request, 'motion/challenge3.html', {'form': form, 'error': str(e)})
    else:
        form = Challenge3Form()
    return render(request, 'motion/challenge3.html', {'form': form})

########### - Challenge 4 - ###########
def max_range_angle(u, h, g):
    theta_max = np.arcsin(1 / np.sqrt(2 + 2 * g * h / u**2))
    return np.degrees(theta_max)

def range_angle(u, angle, g, h):
    angle_rad = np.radians(angle)
    range = (u**2 / g) * (np.sin(angle_rad) * np.cos(angle_rad) + np.cos(angle_rad) * np.sqrt(np.sin(angle_rad)**2 + 2 * g * h / u**2))
    return range

def projectile_motion_4(u, angle, g, h=0):
    angle = np.radians(angle)
    initial_x = u * np.cos(angle)
    initial_y = u * np.sin(angle)
    
    flight_time = (initial_y + np.sqrt(initial_y**2 + 2 * g * h)) / g
    
    T = np.linspace(0, flight_time, num=500)
    
    x = initial_x * T
    y = initial_y * T - 0.5 * g * T**2
    
    return x, y

def plot_projectile_4(u, h, g, theta):
    max_angle = max_range_angle(u, h, g)

    x_theta, y_theta = projectile_motion_4(u, theta, g, h)
    x_max, y_max = projectile_motion_4(u, max_angle, g, h)

    plt.figure(figsize=(10, 5))
    plt.plot(x_theta, y_theta, 'r-', label=f'Trajectory for Theta = ({theta:.1f}°)')
    plt.plot(x_max, y_max, 'b-', label=f'Trajectory for Angle With Max Range = ({max_angle:.1f}°)')

    plt.title('Projectile Trajectories')
    plt.xlabel('Horizontal Distance')
    plt.ylabel('Vertical Distance')
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.grid(True)
    plt.legend()
    plt.show()

def challenge4_view(request):
    if request.method == 'POST':
        form = Challenge4Form(request.POST)
        if form.is_valid():
            theta = form.cleaned_data['theta']
            g = form.cleaned_data['g']
            u = form.cleaned_data['u']
            h = form.cleaned_data['h']

            plot_projectile_4(u, h, g, theta)

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            buf.close()

            return render(request, 'motion/challenge4.html', {'form': form, 'image_base64': image_base64})
    else:
        form = Challenge4Form()
    return render(request, 'motion/challenge4.html', {'form': form})

########### - Challenge 5 - ###########

def plot_projectile_5(u, h, X, Y, g):
    min_u = minimum_velocity(X, Y, g)
    
    # Minimum velocity trajectory
    min_angle = np.degrees(np.arctan((Y + np.sqrt(X**2 + Y**2)) / X))
    x_min_u, y_min_u = projectile_motion_4(min_u, min_angle, g)
    
    try:
        # Calculate high and low angles
        high_angle, low_angle = calculate_trajectory_angles(X, Y, min_u, g)
        
        # High ball trajectory
        x_high, y_high = projectile_motion_4(min_u, high_angle, g)
        
        # Low ball trajectory
        x_low, y_low = projectile_motion_4(min_u, low_angle, g)
        
        # Adjust all trajectories to ensure they pass through the target
        x_min_u, y_min_u = adjust_trajectory_to_target(x_min_u, y_min_u, X, Y)
        x_high, y_high = adjust_trajectory_to_target(x_high, y_high, X, Y)
        x_low, y_low = adjust_trajectory_to_target(x_low, y_low, X, Y)
        
        # Bounding parabola
        x_range = np.linspace(0, X, 500)
        y_bounding = (min_u**2 / (2 * g)) - (g / (2 * min_u**2)) * x_range**2
        
        # Maximum range trajectory
        max_angle = max_range_angle(u, h, g)
        x_max, y_max = projectile_motion_4(u, max_angle, g, h)
        
        # Plot the trajectories
        plt.figure(figsize=(10, 5))
        plt.plot(x_min_u, y_min_u, 'grey', label='Minimum Velocity')
        plt.plot(x_high, y_high, 'blue', label='High Ball')
        plt.plot(x_low, y_low, 'red', label='Low Ball')
        plt.plot(x_range, y_bounding, 'green', label='Bounding Parabola')
        plt.plot(x_max, y_max, 'purple', label='Maximum Range')
        plt.plot(X, Y, 'yo', label='Target', markersize=10)
        
        plt.title('Projectile Trajectories')
        plt.xlabel('Horizontal Distance')
        plt.ylabel('Vertical Distance')
        plt.ylim(bottom=0)
        plt.xlim(left=0)
        plt.grid(True)
        plt.legend()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        
        return image_base64
    except ValueError as e:
        return str(e)

def challenge5_view(request):
    if request.method == 'POST':
        form = Challenge5Form(request.POST)
        if form.is_valid():
            u = form.cleaned_data['u']
            h = form.cleaned_data['h']
            X = form.cleaned_data['X']
            Y = form.cleaned_data['Y']
            g = form.cleaned_data['g']

            image_base64 = plot_projectile_5(u, h, X, Y, g)
            return render(request, 'motion/challenge5.html', {'form': form, 'image_base64': image_base64})
    else:
        form = Challenge5Form()
    return render(request, 'motion/challenge5.html', {'form': form})

########### - Challenge 6 - ###########

def trajectory_length(x, y):
    dx = np.diff(x)
    dy = np.diff(y)
    distance = np.sum(np.sqrt(dx**2 + dy**2))
    return distance

def plot_projectile_6(u, h, X, Y, g):
    min_u = minimum_velocity(X, Y, g)
    
    # Minimum velocity trajectory
    min_angle = np.degrees(np.arctan((Y + np.sqrt(X**2 + Y**2)) / X))
    x_min_u, y_min_u = projectile_motion_4(min_u, min_angle, g)
    
    try:
        # Calculate high and low angles
        high_angle, low_angle = calculate_trajectory_angles(X, Y, min_u, g)
        
        # High ball trajectory
        x_high, y_high = projectile_motion_4(min_u, high_angle, g)
        
        # Low ball trajectory
        x_low, y_low = projectile_motion_4(min_u, low_angle, g)
        
        # Adjust all trajectories to ensure they pass through the target
        x_min_u, y_min_u = adjust_trajectory_to_target(x_min_u, y_min_u, X, Y)
        x_high, y_high = adjust_trajectory_to_target(x_high, y_high, X, Y)
        x_low, y_low = adjust_trajectory_to_target(x_low, y_low, X, Y)
        
        # Bounding parabola
        x_range = np.linspace(0, X, 500)
        y_bounding = (min_u**2 / (2 * g)) - (g / (2 * min_u**2)) * x_range**2
        
        # Maximum range trajectory
        max_angle = max_range_angle(u, h, g)
        x_max, y_max = projectile_motion_4(u, max_angle, g, h)
        
        # Compute lengths of trajectories
        min_u_length = trajectory_length(x_min_u, y_min_u)
        high_length = trajectory_length(x_high, y_high)
        low_length = trajectory_length(x_low, y_low)
        max_range_length = trajectory_length(x_max, y_max)
        
        # Plot the trajectories
        plt.figure(figsize=(10, 5))
        plt.plot(x_min_u, y_min_u, 'grey', label=f'Minimum Velocity (Length: {min_u_length:.2f} m)')
        plt.plot(x_high, y_high, 'blue', label=f'High Ball (Length: {high_length:.2f} m)')
        plt.plot(x_low, y_low, 'red', label=f'Low Ball (Length: {low_length:.2f} m)')
        plt.plot(x_range, y_bounding, 'green', label='Bounding Parabola')
        plt.plot(x_max, y_max, 'purple', label=f'Maximum Range (Length: {max_range_length:.2f} m)')
        plt.plot(X, Y, 'yo', label='Target', markersize=10)
        
        plt.title('Projectile Trajectories')
        plt.xlabel('Horizontal Distance')
        plt.ylabel('Vertical Distance')
        plt.ylim(bottom=0)
        plt.xlim(left=0)
        plt.grid(True)
        plt.legend()
        plt.show()

        # Save plot to a BytesIO object
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()

        return image_base64
    except ValueError as e:
        return str(e)

def challenge6_view(request):
    if request.method == 'POST':
        form = Challenge6Form(request.POST)
        if form.is_valid():
            u = form.cleaned_data['initial_velocity']
            h = form.cleaned_data['launch_height']
            X = form.cleaned_data['x_target']
            Y = form.cleaned_data['y_target']
            g = form.cleaned_data['gravity']
            
            try:
                image_base64 = plot_projectile_6(u, h, X, Y, g)
                return render(request, 'motion/challenge6.html', {'form': form, 'image_base64': image_base64})
            except Exception as e:
                return render(request, 'motion/challenge6.html', {'form': form, 'error': str(e)})
    else:
        form = Challenge6Form()
    return render(request, 'motion/challenge6.html', {'form': form})

########### - Challenge 7 - ###########

def projectile_motion_7(u, angle, g, h=0):
    angle = np.radians(angle)
    initial_x = u * np.cos(angle)
    initial_y = u * np.sin(angle)

    flight_time = (initial_y + np.sqrt(initial_y**2 + 2 * g * h)) / g
    T = np.linspace(0, flight_time, num=500)

    x = initial_x * T
    y = h + initial_y * T - 0.5 * g * T**2

    return x, y, T

def range_vs_time(u, angle, g):
    angle_rad = np.radians(angle)
    initial_x = u * np.cos(angle_rad)
    initial_y = u * np.sin(angle_rad)

    flight_time = (initial_y + np.sqrt(initial_y**2)) / g
    T = np.linspace(0, flight_time, num=500)

    range_t = initial_x * T

    return T, range_t

def turning_points(u, theta, g):
    theta = np.radians(theta)
    sin_theta = np.sin(theta)
    critical_times = []

    if sin_theta >= 2 * np.sqrt(2) / 3:
        t1 = (3 * u / (2 * g)) * (sin_theta + np.sqrt(sin_theta**2 - 8 / 9))
        t2 = (3 * u / (2 * g)) * (sin_theta - np.sqrt(sin_theta**2 - 8 / 9))
        critical_times = [t1, t2]

    return critical_times

def plot_trajectories(u, g):
    angles = [30, 45, 60, 70.5, 78, 85]

    plt.figure(figsize=(10, 5))

    # Plot Range vs Time
    plt.subplot(1, 2, 1)
    for angle in angles:
        T, range_t = range_vs_time(u, angle, g)
        plt.plot(T, range_t, label=f'{angle}°')

        critical_times = turning_points(u, angle, g)
        for t in critical_times:
            range_at_t = u * np.cos(np.radians(angle)) * t
            plt.plot(t, range_at_t, 'o')

    plt.title('Range vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Range (m)')
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.legend()
    plt.grid(True)

    # Plot Y vs X
    plt.subplot(1, 2, 2)
    for angle in angles:
        x, y, T = projectile_motion_7(u, angle, g)
        plt.plot(x, y, label=f'{angle}°')
        plt.plot(x[-1], y[-1], 'ro')  # Mark the end point

    plt.title('Y vs X')
    plt.xlabel('Distance (m)')
    plt.ylabel('Height (m)')
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Save plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    return image_base64

def challenge7_view(request):
    if request.method == 'POST':
        form = Challenge7Form(request.POST)
        if form.is_valid():
            u = form.cleaned_data['initial_velocity']
            g = form.cleaned_data['gravity']

            try:
                image_base64 = plot_trajectories(u, g)
                return render(request, 'motion/challenge7.html', {'form': form, 'image_base64': image_base64})
            except Exception as e:
                return render(request, 'motion/challenge7.html', {'form': form, 'error': str(e)})
    else:
        form = Challenge7Form()
    return render(request, 'motion/challenge7.html', {'form': form})

########### - Challenge 8 - ###########

def verlet_trajectory(N, C, g, dt, h, theta, u):
    theta = np.radians(theta)
    nbounce = 0
    n = 0
    
    t = [0]
    x = [0]
    y = [h]
    vy = [u * np.sin(theta)]
    vx = [u * np.cos(theta)]
    
    while nbounce <= N:
        ax = 0
        ay = -g
        
        x_next = x[n] + vx[n] * dt + 0.5 * ax * dt**2
        y_next = y[n] + vy[n] * dt + 0.5 * ay * dt**2
        
        aax = 0
        aay = -g
        
        vx_next = vx[n] + 0.5 * (ax + aax) * dt
        vy_next = vy[n] + 0.5 * (ay + aay) * dt
        
        t_next = t[n] + dt
        
        if y_next < 0:
            y_next = 0
            vy_next = -C * vy_next
            nbounce += 1
        
        x.append(x_next)
        y.append(y_next)
        vx.append(vx_next)
        vy.append(vy_next)
        t.append(t_next)
        
        n += 1
    
    return np.array(t), np.array(x), np.array(y)

def plot_trajectory(N, C, g, dt, h, theta, u):
    t, x, y = verlet_trajectory(N, C, g, dt, h, theta, u)
    
    plt.figure(figsize=(10, 5))
    
    # Plot Y vs X (Trajectory)
    plt.subplot(1, 2, 1)
    plt.plot(x, y, label=f'Trajectory with {N} bounces')
    plt.xlabel('Horizontal Distance (m)')
    plt.ylabel('Vertical Distance (m)')
    plt.title('Projectile Trajectory with Bounces')
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Save plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    return image_base64

def challenge8_view(request):
    if request.method == 'POST':
        form = Challenge8Form(request.POST)
        if form.is_valid():
            N = form.cleaned_data['N']
            C = form.cleaned_data['C']
            g = form.cleaned_data['g']
            dt = form.cleaned_data['dt']
            h = form.cleaned_data['h']
            theta = form.cleaned_data['theta']
            u = form.cleaned_data['u']

            try:
                image_base64 = plot_trajectory(N, C, g, dt, h, theta, u)
                return render(request, 'motion/challenge8.html', {'form': form, 'image_base64': image_base64})
            except Exception as e:
                return render(request, 'motion/challenge8.html', {'form': form, 'error': str(e)})
    else:
        form = Challenge8Form()
    return render(request, 'motion/challenge8.html', {'form': form})

########### - Challenge 9 - ###########
def projectile_motion_9(u, theta, g, dt, t_max):
    theta = np.radians(theta)
    t = np.arange(0, t_max, dt)
    
    x = u * np.cos(theta) * t
    y = u * np.sin(theta) * t - 0.5 * g * t**2
    
    return t, x, y

def verlet_trajectory_9(u, theta, g, dt, t_max, Cd, rho, A, m):
    theta = np.radians(theta)
    k = 0.5 * Cd * rho * A / m
    
    t = [0]
    x = [0]
    y = [0]
    vx = [u * np.cos(theta)]
    vy = [u * np.sin(theta)]
    
    while t[-1] < t_max and y[-1] >= 0:
        V = np.sqrt(vx[-1]**2 + vy[-1]**2)
        ax = - (vx[-1] / V) * k * V**2
        ay = -g - (vy[-1] / V) * k * V**2
        
        x_next = x[-1] + vx[-1] * dt + 0.5 * ax * dt**2
        y_next = y[-1] + vy[-1] * dt + 0.5 * ay * dt**2
        
        vx_next = vx[-1] + ax * dt
        vy_next = vy[-1] + ay * dt
        
        t_next = t[-1] + dt
        
        x.append(x_next)
        y.append(y_next)
        vx.append(vx_next)
        vy.append(vy_next)
        t.append(t_next)
        
        if y_next < 0:
            break
    
    return np.array(t), np.array(x), np.array(y)

def plot_trajectories(u, theta, g, dt, t_max, Cd, rho, A, m):
    t1, x1, y1 = projectile_motion_9(u, theta, g, dt, t_max)
    t2, x2, y2 = verlet_trajectory_9(u, theta, g, dt, t_max, Cd, rho, A, m)
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(x1, y1, label='No Drag')
    plt.plot(x2, y2, label='With Drag')
    plt.xlabel('Horizontal Distance (m)')
    plt.ylabel('Vertical Distance (m)')
    plt.title('Projectile Trajectory')
    plt.ylim(bottom=0) 
    plt.xlim(left=0)
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(t1, x1, label='No Drag')
    plt.plot(t2, x2, label='With Drag')
    plt.xlabel('Time (s)')
    plt.ylabel('Range (m)')
    plt.title('Range vs Time')
    plt.ylim(bottom=0) 
    plt.xlim(left=0)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Save plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    return image_base64

def challenge9_view(request):
    if request.method == 'POST':
        form = Challenge9Form(request.POST)
        if form.is_valid():
            u = form.cleaned_data['u']
            theta = form.cleaned_data['theta']
            g = form.cleaned_data['g']
            dt = form.cleaned_data['dt']
            t_max = form.cleaned_data['t_max']
            Cd = form.cleaned_data['Cd']
            rho = form.cleaned_data['rho']
            A = form.cleaned_data['A']
            m = form.cleaned_data['m']

            try:
                image_base64 = plot_trajectories(u, theta, g, dt, t_max, Cd, rho, A, m)
                return render(request, 'motion/challenge9.html', {'form': form, 'image_base64': image_base64})
            except Exception as e:
                return render(request, 'motion/challenge9.html', {'form': form, 'error': str(e)})
    else:
        form = Challenge9Form()
    return render(request, 'motion/challenge9.html', {'form': form})

########### - Extension 1 - ###########

g = 9.81  # Default Gravity (m/s²)
Cd = 0.47  # Default Drag Coefficient
rho0 = 1.225  # Default Air Density at Sea Level (kg/m³)
A = 0.01  # Default Cross Sectional Area (m²)
m = 1  # Default Mass (kg)
R = 287  # Gas Constant (J/(kg·K))
T = 288  # Temperature (K)
dt = 0.01  # Default Time Step (s)
t_max = 10  # Default Max Time (s)

def projectile_motion_ext_1(u, theta, dt, t_max):
    theta = np.radians(theta)
    t = np.arange(0, t_max, dt)
    
    x = u * np.cos(theta) * t
    y = u * np.sin(theta) * t - 0.5 * g * t**2
    
    return t, x, y

def verlet_trajectory_ext_1(u, theta, dt, t_max, Cd, rho, A, m):
    theta = np.radians(theta)
    k0 = 0.5 * Cd * A / m
    
    t = [0]
    x = [0]
    y = [0]
    vx = [u * np.cos(theta)]
    vy = [u * np.sin(theta)]

    while t[-1] < t_max and y[-1] >= 0:
        V = np.sqrt(vx[-1]**2 + vy[-1]**2)
        rho = rho0 * np.exp(-y[-1] * g / (R * T))
        k = k0 * rho
        
        ax = - (vx[-1] / V) * k * V**2
        ay = -g - (vy[-1] / V) * k * V**2
        
        x_next = x[-1] + vx[-1] * dt + 0.5 * ax * dt**2
        y_next = y[-1] + vy[-1] * dt + 0.5 * ay * dt**2
        
        vx_next = vx[-1] + ax * dt
        vy_next = vy[-1] + ay * dt
        
        t_next = t[-1] + dt
        
        x.append(x_next)
        y.append(y_next)
        vx.append(vx_next)
        vy.append(vy_next)
        t.append(t_next)
        
        if y_next < 0:
            break
    
    return np.array(t), np.array(x), np.array(y)

def plot_trajectories_ext_1(u, theta, dt, t_max, Cd, rho, A, m):
    t1, x1, y1 = projectile_motion_ext_1(u, theta, dt, t_max)
    t2, x2, y2 = verlet_trajectory_ext_1(u, theta, dt, t_max, Cd, rho, A, m)
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(x1, y1, label='No Drag')
    plt.plot(x2, y2, label='With Drag (Variable Density)')
    plt.xlabel('Horizontal Distance (m)')
    plt.ylabel('Vertical Distance (m)')
    plt.title('Projectile Trajectory')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(t1, x1, label='No Drag')
    plt.plot(t2, x2, label='With Drag (Variable Density)')
    plt.xlabel('Time (s)')
    plt.ylabel('Range (m)')
    plt.title('Range vs Time')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save plot to a BytesIO object and encode to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    
    return img_base64

def extension1_view(request):
    if request.method == 'POST':
        form = Extension1Form(request.POST)
        if form.is_valid():
            u = form.cleaned_data['u']
            theta = form.cleaned_data['theta']
            Cd = form.cleaned_data['Cd']
            rho = form.cleaned_data['rho']
            A = form.cleaned_data['A']
            m = form.cleaned_data['m']
            dt = form.cleaned_data['dt']
            t_max = form.cleaned_data['t_max']

            try:
                # Encode the plot as base64
                image_base64 = plot_trajectories_ext_1(u, theta, dt, t_max, Cd, rho, A, m)

                return render(request, 'motion/extension1.html', {'form': form, 'image_base64': image_base64})

            except Exception as e:
                return render(request, 'motion/extension1.html', {'form': form, 'error': str(e)})

    else:
        form = Extension1Form()

    return render(request, 'motion/extension1.html', {'form': form})

########### - Extension 2 - ###########

def projectile_motion_with_rotation(u, theta, phi, omega, R, g, dt=0.01, max_time=100):
    theta = np.radians(theta)
    phi = np.radians(phi)
    
    vx = u * np.cos(theta) * np.cos(phi)
    vy = u * np.cos(theta) * np.sin(phi)
    vz = u * np.sin(theta)

    lat = 0
    lon = 0
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)
    
    trajectory_x = [x]
    trajectory_y = [y]
    trajectory_z = [z]
    latitudes = [np.degrees(lat)]
    longitudes = [np.degrees(lon)]
    
    t = 0
    while t < max_time:
        x += vx * dt
        y += vy * dt
        z += vz * dt - 0.5 * g * dt**2

        vz -= g * dt

        r = np.sqrt(x**2 + y**2 + z**2)
        lat = np.arcsin(z / r)
        lon = np.arctan2(y, x) + omega * dt

        trajectory_x.append(x)
        trajectory_y.append(y)
        trajectory_z.append(z)
        latitudes.append(np.degrees(lat))
        longitudes.append(np.degrees(lon))

        if z <= 0:
            break

        t += dt

    return latitudes, longitudes, trajectory_x, trajectory_y, trajectory_z

def plot_projectile_ext_2(u, theta, phi):
    omega = 7.2921159e-5
    R = 6371e3
    g = 9.81

    latitudes, longitudes, traj_x, traj_y, traj_z = projectile_motion_with_rotation(u, theta, phi, omega, R, g)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(traj_x, traj_y, traj_z, label='Projectile Trajectory')
    ax.scatter(traj_x[-1], traj_y[-1], traj_z[-1], color='red', label='Impact Point')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.legend()
    plt.show()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return image_base64, latitudes[-1], longitudes[-1]

def extension2_view(request):
    if request.method == 'POST':
        form = Extension2Form(request.POST)
        if form.is_valid():
            u = form.cleaned_data['u']
            theta = form.cleaned_data['theta']
            phi = form.cleaned_data['phi']
            dt = form.cleaned_data['dt']
            max_time = form.cleaned_data['max_time']

            try:
                image_base64, landing_lat, landing_lon = plot_projectile_ext_2(u, theta, phi)
                context = {
                    'form': form,
                    'image_base64': image_base64,
                    'landing_lat': landing_lat,
                    'landing_lon': landing_lon
                }
                return render(request, 'motion/extension2.html', context)
            except Exception as e:
                return render(request, 'motion/extension2.html', {'form': form, 'error': str(e)})
    else:
        form = Extension2Form()

    return render(request, 'motion/extension2.html', {'form': form})