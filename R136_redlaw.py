import numpy as np
import scipy.interpolate as interpolate

def exctinction_R136(waves,Rv):
    """
    R5495 (= approx Rv) dependent extinction law tailored to the cluster R136
    in the Large Magellanic Cloud, where a strong gradient in R5495 values is
    observed.
    - Shape of the optical and NIR part as in Maiz-Apellaniz et al. (2014),
      but parameterised in a way similar to that of Fitzpatrick (1999).
    - UV part of the curve tailored to R136.
    [Input]
    - waves [numpy array]: wavelength range Angstrom
    - Rv [float]: monochromatic total-to-relative extinction R_5495:
      R5495 = A(lam=5495)/(A(lam=4405)-A(lam=5495))
    [Output]:
    - Alam_Av [numpy array]: normalised extinction as a function of
      wavelength, that is, A(lambda)/A_5495.
      Note that (in broadband equivalents): A(lambda)/A(V) = curve / Rv, with
      curve = k(lambda-V) + Rv = E(lambda-V)/E(B-V) + Rv = Alam / E(B-V)
      This expression is used in e.g. Fitzpatrick et al. (2007).
    """

    # Parameters of the UV part of the extinction curve
    c2 = 1.30                # R136 average
    c1 =  2.030 - 3.007*c2   # As in Fitzpatrick (1999)
    c3 = 1.463               # Bump parameters from Gordon et al. (2003).
    c4 = 0.09                # R136 average
    c5 = 5.9                 # As in Fitzpatrick (1999)
    x0 = 4.558               # Bump parameters from Gordon et al. (2003).
    gamma = 0.945            # Bump parameters from Gordon et al. (2003).

    # Defining inverse wavelength range and the curve components
    xx = 10000./ np.array(waves)
    curve = xx*0.
    xcutuv = 10000.0/2700.0
    xspluv = 10000.0/np.array([2700.0,2600.0])
    iuv = np.where(xx >= xcutuv)[0]
    iopir = np.where(xx < xcutuv)[0]
    if (len(iuv) > 0):
        xuv = np.concatenate((xspluv,xx[iuv]))
    else:
        xuv = xspluv

    # UV part of the curve
    yuv = c1  + c2*xuv
    yuv = yuv + c3*xuv**2/((xuv**2-x0**2)**2 +(xuv*gamma)**2)
    yuv = yuv + c4*(0.5392*(np.maximum(xuv,c5)-c5)**2 +
        0.05644*(np.maximum(xuv,c5)-c5)**3) + Rv
    yspluv  = yuv[0:2]
    if (len(iuv) > 0):
        curve[iuv] = yuv[2::]

    # Optical and NIR spline points
    spline_arr = 10000.0/np.array([26500.0, 18000.0, 12200.0, 10000.0, 8696.0,
        5495.0, 4670.0, 4405.0, 4110.0, 3704.0, 3304.0])
    xsplopir = np.concatenate(([0],spline_arr))
    ysplopir = np.array((np.polyval([-0.1097 ,0.1195][::-1], Rv),
            np.polyval([-0.2046 ,0.2228][::-1], Rv),
            np.polyval([-0.3826 ,0.4167][::-1], Rv),
            np.polyval([-0.5270 ,0.5740][::-1], Rv),
            np.polyval([-0.6392 ,0.7147][::-1], Rv),
            np.polyval([-0.0002 ,1.0000][::-1], Rv),
            np.polyval([0.7455 ,1.0023][::-1], Rv),
            np.polyval([1.0004 ,1.0000][::-1], Rv),
            np.polyval([1.3149 ,0.9887][::-1], Rv),
            np.polyval([1.7931 ,0.9661][::-1], Rv),
            np.polyval([2.2580 ,0.9689][::-1], Rv)))


    ysplopir = np.concatenate((np.array([0]),ysplopir))

    # If the wavelength range covers optical/NIR, interpolate from UV to optical
    if (len(iopir) > 0):
        tck=interpolate.splrep(np.concatenate((xsplopir,xspluv)),
            np.concatenate((ysplopir,yspluv)),k=3)
        curve[iopir] = interpolate.splev(xx[iopir], tck)

    Alam_Av = curve/Rv

    return Alam_Av
