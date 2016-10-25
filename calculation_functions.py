import numpy as np
import math





def vincenty_inverse(point1, point2, miles=False):
    """
    Vincenty's formula (inverse method) to calculate the distance (in
    kilometers or miles) between two points on the surface of a spheroid

    Doctests:
    >>> vincenty((0.0, 0.0), (0.0, 0.0))  # coincident points
    0.0
    >>> vincenty((0.0, 0.0), (0.0, 1.0))
    111.319491
    >>> vincenty((0.0, 0.0), (1.0, 0.0))
    110.574389
    >>> vincenty((0.0, 0.0), (0.5, 179.5))  # slow convergence
    19936.288579
    >>> vincenty((0.0, 0.0), (0.5, 179.7))  # failure to converge
    >>> boston = (42.3541165, -71.0693514)
    >>> newyork = (40.7791472, -73.9680804)
    >>> vincenty(boston, newyork)
    298.396057
    >>> vincenty(boston, newyork, miles=True)
    185.414657

    Source: Maurycy Pietrzak
    """

    # WGS 84
    a = 6378137  # meters
    f = 1 / 298.257223563
    b = 6356752.314245  # meters; b = (1 - f)a

    MILES_PER_KILOMETER = 0.621371

    MAX_ITERATIONS = 200
    CONVERGENCE_THRESHOLD = 1e-12  # .000,000,000,001

    # short-circuit coincident points
    if point1[0] == point2[0] and point1[1] == point2[1]:
        return 0.0

    U1 = math.atan((1 - f) * math.tan(math.radians(point1[0])))
    U2 = math.atan((1 - f) * math.tan(math.radians(point2[0])))
    L = math.radians(point2[1] - point1[1])
    Lambda = L

    sinU1 = math.sin(U1)
    cosU1 = math.cos(U1)
    sinU2 = math.sin(U2)
    cosU2 = math.cos(U2)

    for iteration in range(MAX_ITERATIONS):
        sinLambda = math.sin(Lambda)
        cosLambda = math.cos(Lambda)
        sinSigma = math.sqrt((cosU2 * sinLambda) ** 2 +
                             (cosU1 * sinU2 - sinU1 * cosU2 * cosLambda) ** 2)
        if sinSigma == 0:
            return 0.0  # coincident points
        cosSigma = sinU1 * sinU2 + cosU1 * cosU2 * cosLambda
        sigma = math.atan2(sinSigma, cosSigma)
        sinAlpha = cosU1 * cosU2 * sinLambda / sinSigma
        cosSqAlpha = 1 - sinAlpha ** 2
        try:
            cos2SigmaM = cosSigma - 2 * sinU1 * sinU2 / cosSqAlpha
        except ZeroDivisionError:
            cos2SigmaM = 0
        C = f / 16 * cosSqAlpha * (4 + f * (4 - 3 * cosSqAlpha))
        LambdaPrev = Lambda
        Lambda = L + (1 - C) * f * sinAlpha * (sigma + C * sinSigma *
                                               (cos2SigmaM + C * cosSigma *
                                                (-1 + 2 * cos2SigmaM ** 2)))
        if abs(Lambda - LambdaPrev) < CONVERGENCE_THRESHOLD:
            break  # successful convergence
    else:
        return None  # failure to converge

    uSq = cosSqAlpha * (a ** 2 - b ** 2) / (b ** 2)
    A = 1 + uSq / 16384 * (4096 + uSq * (-768 + uSq * (320 - 175 * uSq)))
    B = uSq / 1024 * (256 + uSq * (-128 + uSq * (74 - 47 * uSq)))
    deltaSigma = B * sinSigma * (cos2SigmaM + B / 4 * (cosSigma *
                 (-1 + 2 * cos2SigmaM ** 2) - B / 6 * cos2SigmaM *
                 (-3 + 4 * sinSigma ** 2) * (-3 + 4 * cos2SigmaM ** 2)))
    s = b * A * (sigma - deltaSigma)

    s /= 1000  # meters to kilometers
    if miles:
        s *= MILES_PER_KILOMETER  # kilometers to miles

    return round(s, 6)





def compass_bearing(pointA, pointB):
    """
    Calculates the bearing between two points.

    The formulae used is the following:
        θ = atan2(sin(Δlong).cos(lat2),
                  cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))

    :Parameters:
      - `pointA: The tuple representing the latitude/longitude for the
        first point. Latitude and longitude must be in decimal degrees
      - `pointB: The tuple representing the latitude/longitude for the
        second point. Latitude and longitude must be in decimal degrees

    :Returns:
      The bearing in degrees

    :Returns Type:
      float

    Source: jeromer
    """

    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])

    diffLong = math.radians(pointB[1] - pointA[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
            * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing





def dir2cart(data):
    """
    Converts vector directions, in degrees, to cartesian coordinates, in x,y,z (adapted from pmag.py).
    """

    ints = np.ones(len(data)).transpose() # get an array of ones to plug into dec,inc pairs
    data = np.array(data)
    rad = np.pi / 180

    # case if array of vectors
    if len(data.shape) > 1:
        decs = data[:,0] * rad
        incs = data[:,1] * rad

        # take the given lengths, if they exist
        if data.shape[1] == 3:
            ints = data[:,2]

    # case if single vector
    else:
        decs = np.array(data[0]) * rad
        incs = np.array(data[1]) * rad
        if len(data) == 3:
            ints = np.array(data[2])
        else:
            ints = np.array([1])

    x = ints * np.cos(decs) * np.cos(incs)
    y = ints * np.sin(decs) * np.cos(incs)
    z = ints * np.sin(incs)

    cart = np.array([x,y,z]).transpose()

    return cart





def cart2dir(cart):
    """
    Converts cartesian coordinates, in x,y,z, to vector directions (adapted from pmag.py).
    """

    cart = np.array(cart)
    rad = np.pi / 180

    # case if array of vectors
    if len(cart.shape) > 1:
        Xs = cart[:,0]
        Ys = cart[:,1]
        Zs = cart[:,2]

    # case if single vector
    else:
        Xs = cart[0]
        Ys = cart[1]
        Zs = cart[2]

    if np.iscomplexobj(Xs): Xs = Xs.real
    if np.iscomplexobj(Ys): Ys = Ys.real
    if np.iscomplexobj(Zs): Zs = Zs.real

     # calculate resultant vector length
    Rs = np.sqrt(Xs**2 + Ys**2 + Zs**2)

    # calculate declination taking care of correct quadrants (arctan2) and making modulo 360
    Decs = (np.arctan2(Ys,Xs) / rad) % 360

    try:
         # calculate inclination (converting to degrees)
        Incs = np.arcsin(Zs/Rs) / rad
    except:
        print('trouble in cart2dir - most likely division by zero somewhere')
        return np.zeros(3)

    return np.array([Decs,Incs,Rs]).transpose()





def fisher_mean(data):
    """
    Calculate the Fisher mean from a list of [dec,inc] pairs (adapted from pmag.py).
    """

    N = len(data)
    R = 0
    Xbar = [0,0,0]
    X = []
    fpars = {}

    if N < 2:
        return fpars

    X = dir2cart(data)

    for i in range(len(X)):
        for c in range(3):
            Xbar[c] = Xbar[c] + X[i][c]

    for c in range(3):
        R = R + Xbar[c]**2
    R = np.sqrt(R)

    for c in range(3):
        Xbar[c] = Xbar[c] / R

    direction = cart2dir(Xbar)

    fpars['dec'] = direction[0]
    fpars['inc'] = direction[1]
    fpars['n'] = N
    fpars['r'] = R

    if N != R:
        k = (N-1) / (N-R)
        fpars['k'] = k
        csd = 81 / np.sqrt(k)
    else:
        fpars['k'] = 'inf'
        csd = 0

    b = 20**(1/(N-1)) - 1
    a = 1 - b*(N-R)/R

    if a < -1: a = -1

    a95 = np.arccos(a)*180/np.pi
    fpars['alpha95'] = a95
    fpars['csd'] = csd

    if a < 0: fpars['alpha95'] = 180

    return fpars
