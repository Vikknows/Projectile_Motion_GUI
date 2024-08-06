from django import forms

class Challenge1Form(forms.Form):
    angle = forms.FloatField(label='Angle of Projection', min_value=0, max_value=90, initial=45)
    gravity = forms.FloatField(label='Gravity (m/s²)', min_value=0.1, max_value=10, initial=9.81)
    velocity = forms.FloatField(label='Initial Velocity (m/s)', min_value=1, max_value=100, initial=20)
    height = forms.FloatField(label='Height of Projection (m)', min_value=0, max_value=100, initial=0)

class Challenge2Form(forms.Form):
    angle = forms.FloatField(label='Angle (degrees)', min_value=0, max_value=90, initial=45)
    g = forms.FloatField(label='Gravity (m/s²)', min_value=0.1, max_value=20, initial=9.81)
    u = forms.FloatField(label='Initial Velocity (m/s)', min_value=1, max_value=100, initial=20)
    h = forms.FloatField(label='Height of Projection (m)', min_value=0, max_value=100, initial=0)

class Challenge3Form(forms.Form):
    x_target = forms.FloatField(label='X Target (m)', min_value=0, max_value=100, initial=50)
    y_target = forms.FloatField(label='Y Target (m)', min_value=0, max_value=100, initial=25)
    g = forms.FloatField(label='Gravity (m/s²)', min_value=0.1, max_value=20, initial=9.81)
    
class Challenge4Form(forms.Form):
    theta = forms.FloatField(label='Theta (degrees)', min_value=0, max_value=90, initial=45)
    g = forms.FloatField(label='Gravity (m/s²)', min_value=0.1, max_value=20, initial=9.81)
    u = forms.FloatField(label='Initial Velocity (m/s)', min_value=1, max_value=100, initial=50)
    h = forms.FloatField(label='Launch Height (m)', min_value=0, max_value=100, initial=0)

class Challenge5Form(forms.Form):
    u = forms.FloatField(label='Initial Velocity', min_value=0, max_value=100, initial=20)
    h = forms.FloatField(label='Launch Height', min_value=0, max_value=100, initial=0)
    X = forms.FloatField(label='X Target', min_value=0, max_value=100, initial=50)
    Y = forms.FloatField(label='Y Target', min_value=0, max_value=100, initial=25)
    g = forms.FloatField(label='Gravity', min_value=0.1, max_value=20, initial=9.81)

class Challenge6Form(forms.Form):
    initial_velocity = forms.FloatField(label='Initial Velocity', min_value=0, max_value=100, initial=20)
    launch_height = forms.FloatField(label='Launch Height', min_value=0, max_value=100, initial=0)
    x_target = forms.FloatField(label='X Target', min_value=0, max_value=100, initial=50)
    y_target = forms.FloatField(label='Y Target', min_value=0, max_value=100, initial=25)
    gravity = forms.FloatField(label='Gravity', min_value=0.1, max_value=20, initial=9.81)

class Challenge7Form(forms.Form):
    initial_velocity = forms.FloatField(label='Initial Velocity', min_value=0, max_value=100, initial=50)
    gravity = forms.FloatField(label='Gravity', min_value=0.1, max_value=20, initial=9.81)

class Challenge8Form(forms.Form):
    N = forms.IntegerField(label='Number of Bounces', min_value=0, max_value=10, initial=3)
    C = forms.FloatField(label='Restitution Coefficient', min_value=0, max_value=1, initial=0.8)
    g = forms.FloatField(label='Gravity', min_value=0.1, max_value=20, initial=9.81)
    dt = forms.FloatField(label='Time Step', min_value=0.001, max_value=0.1, initial=0.01)
    h = forms.FloatField(label='Launch Height', min_value=0, max_value=100, initial=0)
    theta = forms.FloatField(label='Launch Angle', min_value=0, max_value=90, initial=45)
    u = forms.FloatField(label='Initial Velocity', min_value=0, max_value=100, initial=50)

class Challenge9Form(forms.Form):
    u = forms.FloatField(label='Initial Velocity (m/s)', min_value=0, max_value=100, initial=50)
    theta = forms.FloatField(label='Launch Angle (deg)', min_value=0, max_value=90, initial=45)
    g = forms.FloatField(label='Gravity (m/s²)', min_value=0.1, max_value=20, initial=9.81)
    dt = forms.FloatField(label='Time Step (s)', min_value=0.001, max_value=0.1, initial=0.01)
    t_max = forms.FloatField(label='Max Time (s)', min_value=0.1, max_value=50, initial=10)
    Cd = forms.FloatField(label='Drag Coefficient', min_value=0.1, max_value=2.0, initial=0.47)
    rho = forms.FloatField(label='Air Density (kg/m³)', min_value=0.1, max_value=2.0, initial=1.225)
    A = forms.FloatField(label='Cross Sectional Area (m²)', min_value=0.001, max_value=1.0, initial=0.01)
    m = forms.FloatField(label='Mass (kg)', min_value=0.1, max_value=10, initial=1)

class Extension1Form(forms.Form):
    u = forms.FloatField(label='Initial Velocity (m/s)', min_value=0, max_value=100, initial=50)
    theta = forms.FloatField(label='Launch Angle (deg)', min_value=0, max_value=90, initial=45)
    dt = forms.FloatField(label='Time Step (s)', min_value=0.001, max_value=0.1, initial=0.01)
    t_max = forms.FloatField(label='Max Time (s)', min_value=0.1, max_value=50, initial=10)
    Cd = forms.FloatField(label='Drag Coefficient', min_value=0.1, max_value=2.0, initial=0.47)
    rho = forms.FloatField(label='Air Density (kg/m³)', min_value=0.1, max_value=2.0, initial=1.225)
    A = forms.FloatField(label='Cross Sectional Area (m²)', min_value=0.001, max_value=1.0, initial=0.01)
    m = forms.FloatField(label='Mass (kg)', min_value=0.1, max_value=10, initial=1)
    g = forms.FloatField(label='Gravity (m/s²)', min_value=0.1, max_value=20, initial=9.81)

class Extension2Form(forms.Form):
    u = forms.FloatField(label='Initial Velocity (m/s)', min_value=0)
    theta = forms.FloatField(label='Elevation Angle (degrees)', min_value=0, max_value=90)
    phi = forms.FloatField(label='Azimuth Angle (degrees)', min_value=0, max_value=360)
    dt = forms.FloatField(label='Time Step (s)', min_value=0.001, initial=0.01)
    max_time = forms.FloatField(label='Max Time (s)', min_value=1, initial=100)