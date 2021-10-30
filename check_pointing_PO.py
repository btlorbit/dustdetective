import numpy as np
import math
import spiceypy as spice
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
np.set_printoptions(suppress=True)


def check_pointing(det_et, METAKR):
    # Local parameters
    # METAKR = '/Users/brennanlutkewitte/CSPICE/cspice/doc/html/lessons/Ering_2004/PO_06_2017.tm.txt'
    # spice.furnsh(METAKR)
    scid = 'CASSINI'        # Spacecraft code is -82
    instid = -82790         # 'CASSINI_CDA' frame relative to 'CASSINI_SC_COORD' frame -82790
                            # 'CASSINI_CDA_BASE'    -82792
                            # 'CASSINI_CDA_ART'     -82791
    obs = 'SATURN_BARYCENTER'
    ref = 'SATURNJ2000'     # ID: 1400699, 'SATURNJ2000'
    abcorr = 'NONE'
    bsight = np.array([0, 0, 1])        # boresight vector in CDA frame
    tol = 0
    # # Define the specific E-ring crossing (using first 5x Enceladus fly-by times in 2005).
    # # utc = ['2005 feb 17 03:30:00', '2005 mar 09 09:08:00', '2005 mar 29 20:32:00', '2005 may 21 06:47:00',
    # #      '2005 jul 14 19:55:00'] '2005 MAR 09 23:18:00' CDA-ring angle:'0.51' CDA-RAM '1.26' [deg]

    # constants
    mu = 37.9311879e6   # [km^3/s^2], Saturn's gravitational parameter, G*M
    pi = math.pi
    R_saturn = 60268    # [km], Saturn's equatorial radius, 1 bar level

# CHECK BORESIGHT-CASSINI VELOCITY ANGLE AND RING-PLANE SEPARATION ANGLE FOR DURATION ENCOMPASSING DETECTION EVENT IN QUESTION.
# IF ERROR RAISED DUE TO LACK OF TIME COVERAGE IN KERNEL, CHECK KERNEL, AND USE CKBRIEF TO CHECK KERNEL COVERAGE


    #utc = ['YYYY MMM DD HH:MM:SS.###']      # OR User-input detection time in Cal-UTC string
    # J2000 time at Jan 01 2017: 536500800.000000

    # det_et = 550354185.4943317              # FUNCTION INPUT, ephemeris time at detection
    det_sclkdp = spice.sce2c(-82, det_et)
    det_utc = spice.et2utc(det_et, 'C', 0)
    det_et = spice.str2et(det_utc)
    det_etj2k = spice.str2et(det_utc[:5] + 'JAN 01 00:00:00 TDB')  # Ephemeris time: sec past J2000 at year beg of detection
    det_doy = (det_et-det_etj2k)/(3600*24)     # DOY number of detection

    det_doy_lbl = str(math.ceil(det_doy))
    det_doy_lbl = det_doy_lbl[0:3]
    det_etj2k_days = det_etj2k/(3600*24)         # et past j2000 conv. to days; used to calculate DOY of time array

    # CREATE ET TIME ARRAY TO ATTAIN INST. POINTING FOR TIME AROUND THE DETECTION FOR PLOTTING
    # Set up time window and intervals to provide instrument pointing information before and after
    hr_window = 4   # number of hours before and after detection time in which instrument pointing will be plotted.
    interval = 2    # number of minutes per interval
    etbeg = math.ceil((det_et-(3600*hr_window))/ (60 * interval)) * (60 * interval) + 60
    etend = math.floor((det_et+(3600*hr_window)) / (60 * interval)) * (60 * interval) - 60
    step = int((((etend - etbeg) / 60) / interval))  # calculated total of intervals
    # check etbeg and etbeg calendar times (they are shifted inward by a few min due to rounding to achieve int type)
    # print("etbeg:", spice.et2utc(etbeg, 'D', 3 ), "\n" "etend:", spice.et2utc(etend, 'D', 3 ))

    # Create time array of ephemeris times
    time = np.array([x * np.round((etend - etbeg) / step) + etbeg for x in range(step + 1)])
    # time = np.array(time)
    # create time array of encoded spacecraft clock times (sclkdp) for the get c-matrix pointing routine:
    time_sclkdp = np.array([spice.sce2c(-82, time[i]) for i in range(len(time))])

    # Retrieve Cassini state vector and define separate position and velocity vectors and calc the magnitude of the
    # particle's position.
    [state, ltime] = spice.spkezr(scid, time, ref, abcorr, obs)
    state = np.array(state)
    pos = state[:, 0:3]
    v_cas = state[:, 3:6]
    R_p = np.array([spice.vnorm(pos[i]) for i in range(len(pos))])

    # Calculate the hypothetical particle velocity.
    # For this definition of the ram vector, the colliding particle is assumed to be on a circular prograde orbit and
    # at its orbital zenith, so v_z = 0.
    v_p_mag = [math.sqrt(mu / R_p[i]) for i in range(len(pos))]

    # Solve for particle x velocity component; conditions for sign of x velocity: if y < 0, the circular
    # orbit prograde requires that the x-velocity component (v_p_x) is positive, and negative when y > 0.
    x_p = pos[:, 0]
    y_p = pos[:, 1]
    v_p_x = np.zeros((len(y_p)))
    for i in range(len(y_p)):
        if y_p[i] < 0:
            v_p_x[i] = np.array([v_p_mag[i] * (1 / math.sqrt(1 + ((x_p[i] ** 2) / (y_p[i] ** 2))))])
        else:
            v_p_x[i] = np.array([-v_p_mag[i] * (1 / math.sqrt(1 + ((x_p[i] ** 2) / (y_p[i] ** 2))))])

    # Collect Y and Z components of v_p and combine into single v_p vector.
    v_p_y = np.array([(-x_p[i] * v_p_x[i]) / y_p[i] for i in range(len(v_p_x))])
    v_p_z = np.zeros(len(v_p_y))
    v_p = np.vstack((v_p_x, v_p_y, v_p_z)).T

    # Solve for the ram vector, which is the difference between
    kram_sj2k = np.array(v_cas - v_p)

    # Get transformation matrix for Cassini_CDA to SJ2000.
    pform = np.array([spice.pxform('CASSINI_CDA', ref, time[i]) for i in range(len(time))])

    cmat_array = np.array([spice.ckgp(instid, time_sclkdp[i], tol, ref) for i in range(len(time_sclkdp))],dtype=object)
    cmat= cmat_array[:,0]
    # cmat = [np.array(cmat[i], dtype='float64').T for i in range(len(time_sclkdp))]
    for i in range(len(time_sclkdp)):
        cmat[i] = np.array(cmat[i], dtype=object).T


    # Get boresight vector in SJ2000.
    # bsight _sj2k = np.array([spice.mxv(pform[i], bsight) for i in range(len(time))])

    bsight_sj2k = np.zeros((len(time_sclkdp), 3))
    for i in range(len(time_sclkdp)):
        bsight_sj2k[i] = np.matmul(cmat[i], bsight)

    # Calculate the separation angle between the boresight and ram vectors.
    sep_ram = np.array([spice.convrt(spice.vsep(kram_sj2k[i], bsight_sj2k[i]), 'RADIANS', 'DEGREES')
                        for i in range(len(time))])

    # Calculate the separation angle between the boresight and the ringplane (a vector parallel to the ring plane,
    # equivalent to a vector of just the x and y components of the boresight).
    bsight_xy_sj2k = np.concatenate((bsight_sj2k[:, 0:2], np.zeros(((len(bsight_sj2k)), 1))), axis=1)


    sep_ring = np.zeros((len(time), 1))
    for i in range(len(time)):
        if bsight_sj2k[i, 2] > 0:
            sep_ring[i] = np.array(spice.convrt(spice.vsep(bsight_xy_sj2k[i], bsight_sj2k[i]), 'RADIANS', 'DEGREES'))
        else:
            sep_ring[i] = np.array(-1 * spice.convrt(spice.vsep(bsight_xy_sj2k[i], bsight_sj2k[i]), 'RADIANS', 'DEGREES'))

    # Convert time to day-of-year for plotting;
    doy = (time / (3600 * 24) - (det_etj2k_days))  #  time array conv. to days - days since j2k (2000 JAN 01 12:00:00)


    # # Plot the angle of separation vs time and angular limit of CDA.
    # fig = plt.figure(figsize=(7.5, 4.5), constrained_layout=True)
    # axs = fig.subplots(2, sharex='all')
    # axs[0].plot(doy, sep_ring, linewidth=1, label='CDA pointing')
    # axs[1].plot(doy, sep_ram, linewidth=1)
    # axs[1].hlines(28, doy[0], doy[-1], linestyle=':', label='CDA FOV')               # horizontal demarking CDA FOV
    #
    # axs[0].vlines(det_doy, -100, 180, linestyle=':', color='red', label='Time of detection')   # vertical line demarking detection time
    # axs[1].vlines(det_doy, -5, 180, linestyle=':', color='red')     # vertical line demarking detection time
    #
    # # Fixed formatter
    # axs[0].yaxis.set_major_locator(ticker.MultipleLocator(50))
    # axs[0].yaxis.set_minor_locator(ticker.MultipleLocator(10))
    # axs[1].yaxis.set_major_locator(ticker.MultipleLocator(50))
    # axs[1].yaxis.set_minor_locator(ticker.MultipleLocator(10))
    #
    # ymajors0 = ["", "-50", "0", "50", ""]
    # axs[0].yaxis.set_major_formatter(ticker.FixedFormatter(ymajors0))
    # axs[0].set_xlim(doy[0], doy[-1])
    # axs[0].set_ylim(-90, 90)
    # ymajors1 = ["", "0", "50", "100", "150", ""]
    # axs[1].yaxis.set_major_formatter(ticker.FixedFormatter(ymajors1))
    # axs[1].set_ylim(0, 180)
    #
    # fig.suptitle(' Cassini CDA Pointing \n {}, DOY: {}'.format(det_utc, det_doy_lbl), size=10, weight='bold')
    # plt.xlabel("Time [DOY decimal]", size=10)
    # # ("CDA - RAM", size=9, weight='bold')
    # axs[0].set_rasterized
    # axs[1].set_rasterized
    # axs[1].grid(which='both', color='gray', linewidth=0.25)
    # axs[0].grid(which='both', color='gray', linewidth=0.25)
    # axs[1].set(ylabel="CDA - RAM [\N{DEGREE SIGN}]")
    # axs[0].set(ylabel="CDA - RP [\N{DEGREE SIGN}]")
    # axs[0].yaxis.label.set_size(10)
    # axs[1].yaxis.label.set_size(10)
    #
    # h = []  # initiate handles array
    # l = []  # initiate labels array
    # for ax in axs:
    #     axLine, axLabel = ax.get_legend_handles_labels()
    #     h.extend(axLine)
    #     l.extend(axLabel)
    # axs[0].legend(h, l, bbox_to_anchor=(.9975, .99), fontsize=8, loc='upper right', borderaxespad=0.)

    # axs[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    # h, l = axs.get_legend_handles_labels()
    # axs[1].legend(h, l, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    # plt.show()


    ####################################################################################################################
    # FIND SPECIFIC POINTING OF DETECTION... repeat same process, but for isolated ephemeris time of detection

    # get encoded spacecraft clock time of detection
    det_sclkdp = np.array(spice.sce2c(-82, det_et))
    [state_cda, ltime_cda] = spice.spkezr(scid, det_et, ref, abcorr, obs)
    state_cda = np.array(state)
    pos_det = state[0, 0:3]
    v_cas_det = state[0, 3:6]
    R_p = np.array([spice.vnorm(pos_det)])

    # Calculate the hypothetical particle velocity.
    # For this definition of the ram vector, the colliding particle is assumed to be on a circular prograde orbit and
    # at its orbital zenith, so v_z = 0.
    v_p_mag = math.sqrt(mu / R_p)

    # Solve for particle x velocity component; conditions for sign of x velocity: if y < 0, the circular
    # orbit prograde requires that the x-velocity component (v_p_x) is positive, and negative when y > 0.
    x_p = pos[0, 0]
    y_p = pos[0, 1]
    v_p_x = 0

    if y_p < 0:
        v_p_x = np.array([v_p_mag * (1 / math.sqrt(1 + ((x_p ** 2) / (y_p ** 2))))])
    else:
        v_p_x = np.array([-v_p_mag * (1 / math.sqrt(1 + ((x_p ** 2) / (y_p ** 2))))])

    # Collect Y and Z components of v_p and combine into single v_p vector.
    v_p_y = np.array([(-x_p * v_p_x) / y_p])
    v_p_z = np.zeros(len(v_p_y))
    v_p_det = np.vstack((v_p_x, v_p_y, v_p_z)).T

    # Solve for the ram vector, which is the difference between
    det_kram_sj2k = np.array(v_cas_det - v_p_det).flatten()
    # Get transformation matrix for Cassini_CDA to SJ2000.
    pform = np.array(spice.pxform('CASSINI_CDA', ref, det_et))
    det_cmat_array = np.array([spice.ckgp(instid, det_sclkdp, tol, ref)], dtype=object)
    det_cmat = det_cmat_array[:, 0]
    # cmat = [np.array(cmat[i], dtype='float64').T for i in range(len(time_sclkdp))]
    det_cmat = np.array(det_cmat, dtype=object).T


    # Get boresight vector in SJ2000.
    bsight_sj2k = np.array([spice.mxv(pform, bsight)]).flatten()


    # Calculate the separation angle between the boresight and ram vectors at the detection time.
    det_sep_ram = np.zeros((1))
    det_sep_ram = np.array(spice.convrt(spice.vsep(det_kram_sj2k, bsight_sj2k), 'RADIANS', 'DEGREES'))

    # Calculate the separation angle between the boresight and the ringplane at the detection time (a vector parallel
    # to the ring plane, equivalent to a vector of just the x and y components of the boresight).
    det_sep_ring = np.zeros((1, 1))

    bsight_xy_sj2k = np.hstack((bsight_sj2k[0:2], np.zeros((1))))

    # bsight_xy_sj2k = np.concatenate(bsight_sj2k[0:2], np.zeros((1,1)))
    if bsight_sj2k[2] > 0:
        det_sep_ring = np.array(spice.convrt(spice.vsep(bsight_xy_sj2k, bsight_sj2k), 'RADIANS', 'DEGREES'))
    else:
        det_sep_ring = np.array(-1 * spice.convrt(spice.vsep(bsight_xy_sj2k, bsight_sj2k), 'RADIANS', 'DEGREES'))

    det_sep_ram, det_sep_ring = np.round((det_sep_ram, det_sep_ring), decimals=2)

    return [det_sep_ram, det_sep_ring]


if __name__ == '__main__':
    check_pointing()
