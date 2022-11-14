# Average, Rv dependent extinction law towards the cluster R136 in the LMC
# By Sarah Brands
# Date created: 7 Oct 2022

import numpy as np
import scipy.interpolate as interpolate

def exctinction_R136(waves,Rv):
    """
    Rv dependent extinction law tailored to the cluster R136 in the
    Large Magellanic Cloud, where a strong gradient in Rv values is observed.

    - Optical and NIR part as in Fitzpatrick (1999).
    - UV part of the curve tailored to R136.

    [Input]
    - waves [numpy array]: wavelength range Angstrom
    - Rv [float]: total-to-relative extinction

    [Output]:
    - Alam_Av [numpy array]: normalised extinction as a function of wavelength,
      that is, A(lambda)/A(V).

      Note that A(lambda)/A(V) = curve / Rv, with
      curve = k(lambda-V) + Rv = E(lambda-V)/E(B-V) + Rv = Alam / E(B-V)
      This expression is used in e.g. Fitzpatrick et al. (2007).
    """

    # Parameters of the UV part of the extinction curve
    c2 = 0.78 + 0.11*Rv      # Tailored to R136
    c1 =  2.030 - 3.007*c2   # As in Fitzpatrick (1999)
    c3 = 1.463               # Bump parameters from Gordon et al. (2003).
    c4 = 0.13                # Tailored to R136
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

    # Optical and NIR spline points and anchors as in Fitzpatrick (1999)
    spline_arr = 10000.0/np.array([26500.0,12200.0,6000.0,5470.0,4670.0,4110.0])
    xsplopir = np.concatenate(([0],spline_arr))
    ysplir = np.array([0.0,0.26469,0.82925])*Rv/3.1
    ysplop = np.array((np.polyval([-4.22809e-01,1.0027,2.13572e-04][::-1], Rv),
                     np.polyval([-5.13540e-02,1.00216,-7.35778e-05][::-1], Rv),
                     np.polyval([ 7.00127e-01,1.00184,-3.32598e-05][::-1], Rv),
                     np.polyval([ 1.19456,1.01707,-5.46959e-03,
                        7.97809e-04,-4.45636e-05][::-1], Rv)))
    ysplopir = np.concatenate((ysplir,ysplop))

    # If the wavelength range covers optical/NIR, interpolate from UV to optical
    if (len(iopir) > 0):
        tck=interpolate.splrep(np.concatenate((xsplopir,xspluv)),
            np.concatenate((ysplopir,yspluv)),k=3)
        curve[iopir] = interpolate.splev(xx[iopir], tck)

    Alam_Av = curve/Rv

    return Alam_Av
