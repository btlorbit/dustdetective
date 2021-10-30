from __future__ import print_function
import os
import numpy as np
import math
import spiceypy as spice
import scipy.stats as st
from scipy.integrate import quad
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import matplotlib.backends.backend_pdf
from matplotlib.ticker import LinearLocator, FormatStrFormatter, StrMethodFormatter
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from check_pointing_PO import check_pointing
from make_safe_filename import make_safe_filename
np.set_printoptions(suppress=True)
from datetime import datetime



# TOGGLES AND SETTINGS
# read_file: ensure detection data file title is entered.
# METAKR: ensure the metakernel file path is correct, and that the metakernel has all kernels covering timespan of data.
# for det_sclk in range(): ensure the range is the length of the sclk data in read_file, or a custom value.

time_i = datetime.now()

def sims_PO():
    # Local parameters
    home_path = os.getcwd()                     # set actual present working directory file path
    read_file = 'Rev 278, 280, 281.csv'         # file with detections' sclk and impact speed min/max.
    cas_orbit = 'Rev 278, 280, 281'             # string tag of read_file for naming files and plots.

    # Define string tags for both the saved figure file name and also the plot titles to describe the conditions of the
    # simulation i.e. "constrainedV_allq" for constrained impact speeds and all perifocal distances (unfiltered).
    # This is the only place these tags need to be adjusted (the filters/conditions themselves need to be adjust in
    # the code as desired.
    cond_plottag = 'All Impact Speeds and q > 1 R$_S$'
    cond_filetag = 'bulk'        # ctrl+F tag: tango

    METAKR = home_path + '/kernels/mk/PO_06_2017.tm.txt'
    spice.furnsh(METAKR)
    scid = 'CASSINI'        # Spacecraft code is -82
    instid = -82790         # 'CASSINI_CDA' frame relative to 'CASSINI_SC_COORD' frame -82790


    obs = 'SATURN_BARYCENTER'
    ref = 'SATURNJ2000'                 # ID: 1400699, 'SATURNJ2000'
    abcorr = 'NONE'                     # aberration correction = 'NONE' when calcs find 'real' rather than obsrv'd. pos
    bsight = np.array([0, 0, 1])        # boresight vector in CDA frame
    tol = 0                             # tolerance
    sun = 'SUN'                         # defined for later solar angle calculations.

    # constants
    mu = 3.793120689736e+07   # [km^3/s^2], Saturn's gravitational parameter, G*M
    # src: https://engineering.purdue.edu/people/kathleen.howell.1/Publications/Journals/2009_JSR_YamDavLonHowBuf.pdf
    pi = math.pi
    R_saturn = 60268            # [km], Saturn's equatorial radius, 1 bar level
    vernal_eqnx = [1, 0, 0]     # Saturn vernal equinox as defined as +X axis in SATURNJ2000 frame

    df = pd.read_csv(read_file)                 # Read sclk tick values, min and max speeds for all detections
    df['sclk'] = df['sclk'].map(str)            # convert to string
    sclk = df['sclk'].values                    # define as sclk list
    impact_speeds_min = df['minimum speed'].values   # define array impact speed minimums from csv
    impact_speeds_max = df['maximum speed'].values   # define array impact speed minimums from csv

    # Create array for storing all modeled orbits from all det. in a dataset (i.e Revs 278, 280, 281)
    # (in other words, all detections within on data .csv being read).
    n_var_data = 21                             # number of variables of interest (for .csv)
    data_rev = np.empty((0, n_var_data))

    # CREATE ARRAY FOR STORING DETECTION (AND ITS MODELED ORBITS) SUMMARY DATA           
    n_var_detdata = 21
    det_sum = np.zeros((len(sclk), n_var_detdata))                                       
    det_data = np.empty((0, n_var_detdata))                                              


    # LOOP THROUGH SINGLE DETECTION EVENT SCLK TIMES, MODELING ORBIT SETS FOR EACH (ctrl+F tag: lima)
    for det_sclk in range(len(sclk)):

    # FILTER BASED ON IMPACT VELOCITY RANGE FROM SPECTRA:
        v_ram_min = impact_speeds_min[det_sclk]     # get min impact velocity determined from spectra
        v_ram_max = impact_speeds_max[det_sclk]     # get max impact velocity determined from spectra

        # FOR IMPACT SPEEDS UP TO 25 KM/S, i.e. if v_ram_max > 25: pass
        # FOR IMPACT SPEEDS AT LEAST 50 KM/S, i.e. if v_ram_max <50 or !=50: pass
        # FOR ALL IMPACT SPEEDS, i.e. if v_ram_max>100: pass
        if v_ram_max >100:   # (ctrl+F tag: victor)
            pass
        else:
            # Define detection/event time for modeled orbits
            # et = spice.str2et(utc)                        # convert from UTC string to et
            et = spice.scs2e(-82, sclk[det_sclk])           # convert from SCLK string to et
            utc = spice.et2utc(et, 'C', 0)
            # print("et:", et)
            print(utc, det_sclk+1, 'of', len(sclk))


            # sclk = spice.sce2c(-82, et)           # convert et to SCLK
            # Create file name for saving output data into csv.
            filename = cas_orbit
            filename = "".join(filename)
            filename = make_safe_filename(filename)

            # Get Cassini state at SCLK time of detection followed by state vector, and inclination).
            [state, ltime] = spice.spkezr(scid, et, ref, abcorr, obs)
            state = np.array(state)
            pos = state[0:3]
            v_cas = state[3:6]
            v_cas_mag = np.array(spice.vnorm(v_cas))
            R_pos = np.array(spice.vnorm(pos))
            v_esc = math.sqrt(2*mu/R_pos)
            h_cas = spice.vcrss(pos, v_cas)
            h_cas_mag = spice.vnorm(h_cas)
            i_cas = math.acos(h_cas[2] / h_cas_mag)

            # Get sun-pointing vector as seen from Saturn for later solar angle calculations.
            [sunpos, ltime] = spice.spkpos(sun, et, ref, abcorr, obs)
            sunpos = [sunpos[0], sunpos[1], 0]

            # ESTABLISH RANGE OF RAM VELOCITIES (=-1*v_impact) FOR MODELED IMPACTS:
            #
            # For example, use circular prograde particles and from prograde to polar Cassini orbits/fly-bys.

            v_part_kep = math.sqrt(mu/R_pos)

            # Min is e.g. a collision when their orbits are co-linear at a given instant (could be prograde ring plane orbits)
            # Max is e.g. a vertical/polar trajectory of Cassini colliding with an equatorial/ring plane prograde particle.
            # v_ram_min = np.array(v_cas_mag - v_part_kep)
            # v_ram_max = v_cas_mag           # np.array(v_cas_mag + math.sqrt(2*mu/R_pos)) # including retrograde collision

            # Set predefined number of intervals for every detection such that the step size is equal.
            n_step = 15
            # (option) Define n_step according to a user-input interval size, i.e. 1 or 0.5 km/s.
            # intvl_size = 1.0        # interval size in km/s
            # n_step = int(((v_ram_max-v_ram_min)*(1/intvl_size))+1) # n_step for non-int min/max
            v_sim_mag = np.linspace(v_ram_min, v_ram_max, n_step, True)
            # n_step = int(((math.floor(v_ram_max)-math.ceil(v_ram_min))*(1/intvl_size))+1) # n_step for non-int min/max
            # intvl_range = np.linspace(v_ram_min, v_ram_max, n_step, True)                 # for n_step of non-int min/max
            # v_sim_mag = np.concatenate((v_ram_min, intvl_range, v_ram_max), axis=None)    # for n_step of non-int min/max

            # step through the v_sim_mag values and apply to direction via theta and phi (starting with the boresight)
            # initially, this is in the CDA frame (boresight = [1, 0, 0]), but must be converted to SJ2000.

            # Create theta range of evenly spaced angles (optional); if this is the case, a solid-angle weight factor for the
            # probability of the impact theta angle must be incorporated. See variable "SA_wf".
            # theta_step = 4
            # n_theta_step  = int(28/theta_step + 1)
            # theta = np.linspace(0, (28/180)*pi, n_theta_step, True)       # 0 to 28 deg in 4 deg steps = 8 + 1 values, conv. 28 deg to rad


            dphi = 10                                 # phi step size = 20 deg
            n_dphi = int(((360-dphi)/dphi) + 1)
            phi = np.linspace(0, (2*pi)-(dphi*(pi/180)), n_dphi, True)       # [rad]; 0 to (360-step) deg in 20 deg steps = 17 + 1, given in rad

            # # Create range of theta angles such that each step creates an equal spherical surface element (dtheta*dphi)
            # # This ensures that each impact vector represents an equivalent solid angle to avoid artificial bias of the impact distribution
            n_dtheta = 14         # number of dtheta steps, total number of theta steps incl. boresight zero angle = n_dtheta + 1
            theta_min = 0         # boresight = 0 degrees
            theta_max = (28*pi)/180        # [rad]; max FOV angle from boresight = 28 degrees
            dtheta_0 = math.acos((n_dphi + ((math.cos(theta_max))/(n_dtheta-1)))/(n_dphi + (1/(n_dtheta-1))))  # first angle (or dtheta) from boresight to next theta step
            dA = ((2*pi/n_dphi)*(math.cos(dtheta_0)-math.cos(theta_max)))/(n_dtheta-1)  # surface area of one element (= for all elements)
            theta = np.concatenate((np.array([0, dtheta_0]), np.zeros(n_dtheta-1)))       # initiate theta array
            # Iterate theta angle calculations using preceeding theta angle.
            for i in range(2, n_dtheta+1):
                theta[i] = math.acos(math.cos(theta[i - 1]) - (n_dphi * dA / (2 * pi)))
            n_theta_step = len(theta)

            # # check theta
            #print("theta array (deg):", np.round((theta*180/pi), decimals=2))
            # # check dtheta_0 in degrees, check dA surface element area
            # print("d_theta_0 (deg):", ((dtheta_0*180)/pi))
            # print("dA, surface element area:", dA)

            # Create weight factor for the solid angle differential encompassed by each angular step, which changes by sin(theta).
            # For theta=0, to avoid a zero weighting factor, the equivalent dimension of dA is sin(theta[1]/2).
            # SA_wf = np.empty((len(theta)))
            # for j in range(len(theta)):
            #     if j==0:
            #         SA_wf[0] = math.sin((theta[1] / 2))
            #     elif theta[j]>0:
            #         SA_wf[j] = math.sin(theta[j])
            # SA_wf = SA_wf/sum(SA_wf)


            #   USE FOLLOWING V_RAM STATISTICAL MODELING IF A PROBABILITY DIST IS TO BE APPLIED TO IMPACT VELOCITY
            #
            #
            # Determine or set mean or expected average value of impact velocity;
            # # FOR EXAMPLE, can be assumed to be and calculated as prograde keplerian particle:

            # Solve for particle x velocity component; conditions for sign of x velocity: if y < 0, the circular
            # orbit prograde requires that the x-velocity component (v_p_x) is positive, and negative when y > 0.

            x_p_kep = np.array(pos[0])
            y_p_kep = np.array(pos[1])
            v_p_x_kep = np.zeros(1)

            if y_p_kep < 0:
                v_p_x_kep = v_part_kep * (1 / math.sqrt(1 + ((x_p_kep ** 2) / (y_p_kep ** 2))))
            else:
                v_p_x_kep = -v_part_kep * (1 / math.sqrt(1 + ((x_p_kep ** 2) / (y_p_kep ** 2))))

            # Collect Y and Z components of v_p and combine into single v_p vector.
            v_p_y_kep = (-x_p_kep * v_p_x_kep) / y_p_kep
            v_p_z_kep = np.zeros(1)
            v_p_kep = np.vstack((v_p_x_kep, v_p_y_kep, v_p_z_kep)).T

            # Solve for the ram vector, which is the difference between
            v_ram_kep = v_cas - v_p_kep
            # check Kepler ram velocity
            # print("v_ram_kep:", v_ram_kep)


            # Set up probability distributions for v_ram (v_impact).
            # Set mean v_ram value and FWHM values for a Gaussian distribution for v_ram.
            v_ram_mean = spice.vnorm(v_ram_kep)       # 20.1 is based on CDA data from Rev 278.
            FWHM = 2             # arbitrary value for first iteration
            stdev = FWHM/(2*math.sqrt(np.log(2)))
            # print("v_ram_mean:", np.round(v_ram_mean, decimals=2))
            # print('1*stdev: {:.2f}'.format(stdev), "\n",
            #       '2*stdev: {:.2f}'.format(2*stdev), "\n",
            #       '3*stdev: {:.2f}'.format(3 * stdev))
            #
            # print("mean - 3 stdev:", (v_ram_mean - 3*stdev), "\n",
            #       "mean + 3 stdev", (v_ram_mean))


            # # Define the pdf as a function so it can be integrated with scipy.integrate.quad.
            def normal_dist_function(v_sim_mag):
                pdf_v = st.norm.pdf(v_sim_mag, v_ram_mean, stdev)
                return pdf_v
            pdf_v_norm = normal_dist_function(v_sim_mag)/sum(normal_dist_function(v_sim_mag))


            # Plot the normalized pdf to visualize.

            # fig1, ax = plt.subplots(1, figsize=(6, 4), constrained_layout=True)
            # fig1.suptitle('Impact Speed PDF', size=10, weight='bold')
            # ax.plot(v_sim_mag, pdf_v_norm)
            # # axs[0].set_xlim(v_ram_min, v_ram_max)
            # ax.set_xlabel('Impact Velocities [km/s] \n {}'.format(utc), size=10)
            # ax.set_ylabel('Frequency')
            # ax.grid(which='both', color='gray', linewidth=0.25)
            # # axs[0].set_xlim(v_sim_mag[0], v_sim_mag[len(v_sim_mag)-1])
            # ax.set_ylim(0, 1.1*np.max(pdf_v_norm))
            # ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            # ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
            # # print('min + 1 intvl: {:16.0f}'.format(math.ceil(v_ram_min)))
            # textstr = '\n'.join((
            #     r'# samples = {:.0f}'.format(len(v_sim_mag)),
            #     r'FWHM = {:.2f}'.format(FWHM),
            #     r'$\mu$ = {:.2f} km/s'.format(v_ram_mean),
            #     r'$\sigma$ = {:.2f}'.format(stdev)))
            # # these are matplotlib.patch.Patch properties
            # props1 = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            # # place a text box in upper left in axes coords
            # ax.text(0.79, 0.97, textstr, transform=ax.transAxes, fontsize=8, va='top', bbox=props1)

            #
            #
            #   END STATISTICAL MODEL OF IMPACT VELOCITY PROBABILITY DISTRIBUTION



            # THE POLYFIT FUNCTION SHOULD BE FITTED TO THE NORMALIZED DATASET, THEN THE FUNCTION CAN BE EVALUATED FOR
            # EVERY theta VALUE IN THE THE SIMULATION (CORRESPONDS TO X-INPUT VAL) TO GET CORRESPONDING NORMALIZED P(theta)

            # Take empirical angular sensitivity values, fit a polynomial function, and integrate to normalize into a pdf.
            def polyfit(x, a, b, c, d, e, f, g, h, i, j, k, l, m):
                return a * x ** 12 + b * x ** 11 + c * x ** 10 + d * x ** 9 + e * x ** 8 + f * x ** 7 + g * x ** 6 + h * x \
                       ** 5 + i * x ** 4 + j * x ** 3 + k * x ** 2 + l * x + m

            df = pd.read_csv('CAT_exp_pdf.csv')
            # Get the 'theta angle' column
            theta_empirical = df['theta angle'].values
            a_sens_theta = df['angular sensitivity'].values
            a_sens_theta_norm = df['normalized angular sensitivity'].values
            optimal_prmtrs, _ = curve_fit(polyfit, theta_empirical, a_sens_theta_norm)
            # optimal_prmtrs, _ = curve_fit(polyfit, theta_empirical, a_sens_theta) # for polyfit of empirical data

            # summarize the parameter values
            a, b, c, d, e, f, g, h, i, j, k, l, m = optimal_prmtrs
            # pdf_theta = polyfit(x, a, b, c, d, e, f, g, h, i, j, k, l, m)

            def poly_dist_function(theta_angle, a, b, c, d, e, f, g, h, i, j, k, l, m):
                a, b, c, d, e, f, g, h, i, j, k, l, m = optimal_prmtrs
                pdf_theta_poly = polyfit(theta_angle, a, b, c, d, e, f, g, h, i, j, k, l, m)
                return pdf_theta_poly


            # pdf_theta_integrated, err = quad(poly_dist_function, min(theta_empirical), max(theta_empirical), (a, b, c, d, e, f, g, h, i, j, k, l, m))
            pdf_theta = poly_dist_function(theta_empirical, a, b, c, d, e, f, g, h, i, j, k, l, m)

            #
            # NORMALIZED POLYFIT/PDF DISTRIBUTION FUNCTION OF ANGULAR SENSITIVITY
            #
            # fig1p2, ax = plt.subplots(1, figsize=(6, 4), constrained_layout=True)
            # fig1p2.suptitle('Impact Angle PDF', size=10, weight='bold')
            # ax.plot(theta_empirical, pdf_theta)
            # ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            # ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
            # ax.set_xlabel('Theta Impact Angle [\N{DEGREE SIGN}]', size=10)
            # ax.set_ylabel('Frequency')
            # ax.grid(which='both', color='gray', linewidth=0.25)
            # ax.set_xlim(theta_empirical[0], theta_empirical[len(theta_empirical)-1])
            # ax.set_ylim(0, np.max(pdf_theta)+(0.1*np.max(pdf_theta)))
            # ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            # ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

            #
            # EMPIRCAL CAT SENSITIVITY PLOT WITH OPTIONAL FIT LINE TO SHOW CONCEPT OF POLYFIT
            #
            # fig2, ax = plt.subplots(1, figsize=(6, 4))
            # fig2.suptitle('CAT Angular Sensitivity', size=10, weight='bold')
            # # create a line plot for the mapping function
            # ax.plot(theta_empirical, pdf_theta, '--', color='red', label='fit function')
            # x, y = [theta_empirical, a_sens_theta]
            # ax.scatter(x, y, label='simulation')
            # ax.set_xlim(x[0], x[-1])
            # ax.set_ylim(0, np.max(y)+0.1*np.max(y))
            # ax.set_xlabel('Theta Impact Angle [\N{DEGREE SIGN}]')
            # ax.set_ylabel('Angular Sensitivty [unitless]', labelpad=5)
            # ax.grid(which='both', color='gray', linewidth=0.25)
            # ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            # ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
            # h = []  # initiate handles array
            # l = []  # initiate labels array
            # axLine, axLabel = ax.get_legend_handles_labels()
            # h.extend(axLine)
            # l.extend(axLabel)
            # ax.legend(h, l, bbox_to_anchor=(.9975, .99), fontsize=8, loc='upper right', borderaxespad=0.)


            # Initialize variables for  v_ram (in CDA frame), v_particle (in SJ2000), v_p_ag, angular momentum vector h,
            # the eccentricity vector, and the particle orbital elements: a, e, i, w, nu, lan, and intermediate calculation
            # vectors.
            # n_sim = r'n = {:.0f}'.format(len(v_sim_mag)*len(theta)*len(phi))    # number of modeled orbits string
            v_ram = np.zeros((len(v_sim_mag), len(theta), len(phi), 3))         # impact velocity vector
            v_ram_check = np.empty((0,3))                                       # impact velocity vector check for plot
            v_p = np.zeros((len(v_sim_mag), len(theta), len(phi), 3))           # particle velocity vector
            v_p_mag = np.zeros((len(v_sim_mag), len(theta), len(phi), 1))
            # v_p_sj2k = np.zeros((len(v_sim_mag), len(theta), len(phi), 6))

            state_p = np.zeros((len(v_sim_mag), len(theta), len(phi), 6))
            oscelt_p = np.zeros((len(v_sim_mag), len(theta), len(phi), 8))
            L_s = np.zeros((len(v_sim_mag), len(theta), len(phi), 1))
            lan_s = np.zeros((len(v_sim_mag), len(theta), len(phi), 1))

            elements_p = np.zeros((len(v_sim_mag), len(theta), len(phi), n_var_data))    # array of elements, indexes, and prob
            P_theta = np.zeros((len(v_sim_mag), len(theta), len(phi), 1))         # probability of impact at theta angle
            P_v_sim_mag = np.zeros((len(v_sim_mag), len(theta), len(phi), 1))   # probability of impact with simulated impact v
            P_orbit = np.zeros((len(v_sim_mag), len(theta), len(phi), 1))       # combined probability
            v_ram_mag = np.zeros((len(v_sim_mag), len(theta), len(phi), 1))     # impact speed


            # CREATE MAIN STORAGE ARRAY FOR EACH DETECTION'S OUTPUT ORBITS (data_p) and subsequent arrays for binning
            # for select multi-scatter plots.
            data_p = np.empty((0,n_var_data))
            data_p_i10 = np.empty((0, n_var_data))
            data_p_i20 = np.empty((0, n_var_data))
            data_p_i30 = np.empty((0, n_var_data))
            data_p_i40 = np.empty((0, n_var_data))
            data_p_i45 = np.empty((0, n_var_data))
            data_p_i50 = np.empty((0, n_var_data))
            data_p_i60 = np.empty((0, n_var_data))
            data_p_i70 = np.empty((0, n_var_data))
            data_p_i80 = np.empty((0, n_var_data))
            data_p_i90 = np.empty((0, n_var_data))

            # data arrays for 5 degree bins ('f' for five, i.e. 60f represents i=[60 65])
            data_p_i50f = np.empty((0, n_var_data))
            data_p_i55 = np.empty((0, n_var_data))
            data_p_i60f = np.empty((0, n_var_data))
            data_p_i65 = np.empty((0, n_var_data))
            data_p_i70f = np.empty((0, n_var_data))

            # Call check_pointing function to return (ram separation angle, ring separation angle) and to plot
            # CDA pointing around detection time et, if desired/toggled 'on'.
            ram_sep, ring_sep = check_pointing(et, METAKR)

            # Get the rotation matrix (or DCM transformation from CASSINIA_CDA to SJ2000 at time 'et'. Output is 3x3.
            pform = spice.pxform('CASSINI_CDA', ref, et)

            # Step through v_sim_mag, theta, and phi and produce an array of arrays with new xyz particle velocity components
            # in SJ2000 with corresponding inputs (v_sim_mag, theta, phi) and data of interest.

            for i in range(len(v_sim_mag)):
                for j in range(len(theta)):
                    for k in range(len(phi)):

                        # Establish unit vector of v_ram in CDA frame.
                        v_ram[i, j, k] = [1 * math.sin(theta[j]) * math.cos(phi[k]),
                                          1 * math.sin(theta[j]) * math.sin(phi[k]),
                                          1 * math.cos(theta[j])]
                        v_ram_check = np.append(v_ram_check, v_ram[i, j, k].reshape((1, -1)), axis=0)
                        v_ram[i, j, k] =  v_sim_mag[i]* spice.mxv(pform, v_ram[i,j,k])  # convert v_ram in CDA to SJ2000
                        v_ram_mag[i, j, k]  = spice.vnorm(v_ram[i,j,k])
                        v_p[i, j, k]        = np.array(v_cas - v_ram[i, j, k])

                        state_p[i, j, k] = np.r_[pos, v_p[i,j,k]]   # combine position and v_p into grain state vector
                        oscelt_p[i, j, k] = spice.oscelt(state_p[i,j,k], et, mu) # get osculating elements
                        L_s[i, j, k] = spice.vsep(sunpos,
                                                  vernal_eqnx)  # solar longitude of Saturn w.r.t. Saturn vernal equinox

                        if oscelt_p[i, j, k, 3] >= L_s[i, j, k]:
                            lan_s[i, j, k] = (oscelt_p[i, j, k, 3] - L_s[i,j,k])*180/pi       # LAN w.r.t. solar longitude
                        else:
                            lan_s[i, j, k] = (oscelt_p[i, j, k, 3] - L_s[i,j,k] + (2*pi))*180/pi

                        v_p_mag[i, j, k]    = spice.vnorm(v_p[i,j,k])

                        P_theta[i, j, k]  = (poly_dist_function(theta[j]*180/pi, a, b, c, d, e, f, g, h, i, j, k, l, m)) #*SA_wf[j]
                        P_orbit[i, j, k] = P_theta[i, j, k]  # not considering any probability distribution of impact v's
                        # P_v_sim_mag[i, j, k] = pdf_v_norm[i]
                        # P_orbit[i, j, k]   = P_theta[i,j,k]*P_v_sim_mag[i,j,k]
                        # P_orbit[i, j, k]   = P_theta[i,j,k]   # not considering any probability distribution of impact v's



                        # print("theta angle:", theta[j], "P_theta:", P_theta[i,j,k])

                        # Form array of data of interest for output and further analysis:
                        #   simulated impact speed: v_sim_mag,
                        #   Theta impact angle, phi impact angle,
                        #   simulated particle speed: v_p_mag
                        #   orbital elements of particle: a [km], a [R_s], e, i, omega, lan
                        #   composite relative probability of orbit, P_orbit
                        #   component probabilities: P(theta), and P(v_impact)

                                                # VARIABLE                                              # INDEX
                        elements_p[i, j, k] =   [v_sim_mag[i],                                          # 0 imp. V  [km/s]
                                                theta[j]*180/pi,                                        # 1 theta     [deg]
                                                phi[k]*180/pi,                                          # 2 phi   [deg]
                                                v_p_mag[i,j,k],                                         # 3 part. V [km/s]
                                                oscelt_p[i,j,k,0]/(1-oscelt_p[i, j, k, 1]),             # 4 sem-maj [km]
                                                (oscelt_p[i,j,k,0]/(1-oscelt_p[i, j, k, 1]))/R_saturn,  # 5 sem-maj [R_s]
                                                oscelt_p[i, j, k, 1],                                   # 6 eccentricity
                                                oscelt_p[i, j, k, 2]*180/pi,                            # 7 inclin. [deg]
                                                oscelt_p[i, j, k, 4]*180/pi,                            # 8 arg.p   [deg]
                                                oscelt_p[i, j, k, 3]*180/pi,                            # 9 lnode   [deg]
                                                P_orbit[i,j,k],                                         # 10 total prob.
                                                P_theta[i,j,k],                                         # 11 theta prob
                                                P_v_sim_mag[i,j,k],                                     # 12 velocity prob
                                                (oscelt_p[i, j, k, 4] + oscelt_p[i, j, k, 3] - L_s[i,j,k])*180/pi,
                                                                                    # 13 lperikrone relative to L_s   [deg]
                                                lan_s[i,j,k],                       # 14 lnode relative to L_s [deg]
                                                sclk[det_sclk],                     # 15 SCLK for tracking in output csv
                                                ram_sep,                                                # 16 RAM sep. [deg]
                                                ring_sep,                                               # 17 RP sep [deg]
                                                R_pos/R_saturn,                                         # 18 Cass distance
                                                0,                                                      # 19 temp filler for P_orbit_norm
                                                np.round((pos[2]/R_saturn), decimals=2)]                # 20 RP distance


                        ### OUTPUT FILTERS ###
                        # Remove duplicate boresight impact vectors (leaving 1 combo of theta/phi=0) and optional:
                        # orbits with perikrone distance below 1 R_s.
                        # Then store in data_p (p for particle) arrays.
                        if j == 0 and k != 0 or elements_p[i,j,k,5]*(1-elements_p[i,j,k,6]) <=1:       # ctrl+f tag: foxtrot
                            pass
                        else:
                            data_p = np.append(data_p, elements_p[i,j,k].reshape((1,n_var_data)), axis=0)


                            if elements_p[i, j, k, 7] >= 0 and elements_p[i, j, k, 7] <= 10:
                                data_p_i10 = np.append(data_p_i10, elements_p[i, j, k].reshape((1, n_var_data)), axis=0)
                            elif elements_p[i, j, k, 7] > 10 and elements_p[i, j, k, 7] <= 20:
                                data_p_i20 = np.append(data_p_i20, elements_p[i, j, k].reshape((1, n_var_data)), axis=0)
                            elif elements_p[i, j, k, 7] > 20 and elements_p[i, j, k, 7] <= 30:
                                data_p_i30 = np.append(data_p_i30, elements_p[i, j, k].reshape((1, n_var_data)), axis=0)
                            elif elements_p[i, j, k, 7] > 30 and elements_p[i, j, k, 7] <= 40:
                                data_p_i40 = np.append(data_p_i40, elements_p[i, j, k].reshape((1, n_var_data)), axis=0)
                            elif elements_p[i, j, k, 7] > 40 and elements_p[i, j, k, 7] <= 50:
                                data_p_i50 = np.append(data_p_i50, elements_p[i, j, k].reshape((1, n_var_data)), axis=0)
                            elif elements_p[i, j, k, 7] > 50 and elements_p[i, j, k, 7] <= 60:
                                data_p_i60 = np.append(data_p_i60, elements_p[i, j, k].reshape((1, n_var_data)), axis=0)
                            elif elements_p[i, j, k, 7] > 60 and elements_p[i, j, k, 7] <= 70:
                                data_p_i70 = np.append(data_p_i70, elements_p[i, j, k].reshape((1, n_var_data)), axis=0)
                            elif elements_p[i, j, k, 7] > 70 and elements_p[i, j, k, 7] <= 80:
                                data_p_i80 = np.append(data_p_i80, elements_p[i, j, k].reshape((1, n_var_data)), axis=0)
                            elif elements_p[i, j, k, 7] > 80 and elements_p[i, j, k, 7] <= 90:
                                data_p_i90 = np.append(data_p_i90, elements_p[i, j, k].reshape((1, n_var_data)), axis=0)

                            elif elements_p[i, j, k, 7] > 40 and elements_p[i, j, k, 7] <= 45:
                                data_p_i45 = np.append(data_p_i45, elements_p[i, j, k].reshape((1, n_var_data)), axis=0)
                            elif elements_p[i, j, k, 7] > 50 and elements_p[i, j, k, 7] <= 55:
                                data_p_i50f = np.append(data_p_i50f, elements_p[i, j, k].reshape((1, n_var_data)), axis=0)
                            elif elements_p[i, j, k, 7] > 50 and elements_p[i, j, k, 7] <= 55:
                                data_p_i55 = np.append(data_p_i55, elements_p[i, j, k].reshape((1, n_var_data)), axis=0)
                            elif elements_p[i, j, k, 7] > 55 and elements_p[i, j, k, 7] <= 60:
                                data_p_i60f = np.append(data_p_i60f, elements_p[i, j, k].reshape((1, n_var_data)), axis=0)
                            elif elements_p[i, j, k, 7] > 60 and elements_p[i, j, k, 7] <= 65:
                                data_p_i65 = np.append(data_p_i65, elements_p[i, j, k].reshape((1, n_var_data)), axis=0)
                            elif elements_p[i, j, k, 7] > 65 and elements_p[i, j, k, 7] <= 70:
                                data_p_i70f = np.append(data_p_i70f, elements_p[i, j, k].reshape((1, n_var_data)), axis=0)


            # Accumulate all orbit data for each detection into data_rev array (array for all detections during an
            # entire Cassini revolution and/or all detections within the read file i.e. multiple revs).
            data_p[:,19] = data_p[:,10]/sum(data_p[:,10])     # normalize all probability values accross the modeled detection (det_sclk)
            data_rev = np.append(data_rev, data_p, axis=0)    # store all values 

            n_sim = r'n = {:.0f}'.format(len(data_p))    # number of modeled orbits string

            v_imp_vals = data_p[:,0]
            v_p_vals = data_p[:,3]
            a_vals = data_p[:,5]
            e_vals = data_p[:,6]
            i_vals = data_p[:,7]
            w_vals = data_p[:,8]

            wbar_ls_vals = data_p[:,13]
            lan_vals = data_p[:,14]
            P_vals = data_p[:,19]   # probabilities of modeled grain orbits within a single detection dataset


            # # Plot the orbital shape elements a,e, and i on a 3D plot with color scale for relative probability
            # fig3 = plt.figure(figsize=(6,4))
            # ax3 = fig3.gca(projection='3d')
            # y_ulim = 20
            # y_llim = -20
            # x = e_vals
            # y = a_vals
            # z = i_vals
            # y[y > y_ulim] = np.nan
            # y[y < y_llim] = np.nan
            # color = P_vals
            # fig3.suptitle(r"Bound Orbit Shape and Relative Probability" "\n" r"{0}, SCLK: {1}".format(utc, sclk[det_sclk]),
            #               size=10, weight='bold')
            # ax3.set_xlabel('Eccentricity, e')
            # ax3.set_ylabel('Semi-major Axis, a [$R_S$]')
            # ax3.set_zlabel('Inclination, i [\N{DEGREE SIGN}]')
            # ax3.set_xlim(0, x.max())
            # ax3.set_ylim(y_llim, y_ulim)
            # ax3.set_zlim(0, z.max()+(.10*z.max()))
            # ax3.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
            # text_x = 0.68
            # text_y = 0.82
            # props1 = dict(boxstyle='round', facecolor='wheat', alpha=0.6)        # create textbox style
            # # place textbox with number of samples
            # fig3.text(text_x, text_y, n_sim, fontsize=8, va='top', ha='right')
            # img = ax3.scatter(x, y, z, s=10, c=color, cmap=plt.jet(), alpha=0.5)
            # fig3.colorbar(img)



            fig4, ax4 = plt.subplots(1, figsize=(8.5, 4))
            fig4.suptitle(r"Orbit Shape (a vs. e) Histogram" "\n" r"{0}, SCLK: {1}".format(utc, sclk[det_sclk]),
                          size=10, weight='bold')
            x = a_vals
            y = e_vals
            ax4.set_xlabel('Semi-major Axis, a [$R_S$]')
            ax4.set_ylabel('Eccentricity, e')
            # text coordinates
            text_x = 0.985
            text_y = 0.05
            props1 = dict(boxstyle='round', facecolor='wheat', alpha=0.6)        # create textbox style
            # place textbox with number of samples
            ax4.text(text_x, text_y, n_sim, transform=ax4.transAxes, fontsize=8, va='top', ha='right', bbox=props1)
            x_bins = np.linspace(-15, 15, 20, True)
            # y_bins = np.linspace(0, 3, 20, True)
            if max(y) >=10:
                x_bins = np.linspace(-15, 15, 24, True)
                y_bins = 24
            elif 6 <= max(y) < 10:
                x_bins = np.linspace(-15, 15, 20, True)
                y_bins = 20
            else:
                x_bins = np.linspace(-15, 15, 15, True)
                y_bins = 15
            count, x_edges, y_edges, hist_scale = ax4.hist2d(x, y, bins=[x_bins, y_bins], density=True, weights=P_vals, cmap=plt.cm.jet, rasterized=True)
            fig4.colorbar(hist_scale).set_label('Probability-weighted frequency', rotation=270, labelpad=18)
            props2 = dict(boxstyle='square', facecolor='white')
            textstr = '\n'.join((
                'Cassini Distance: {:.2f} $R_S$'.format(R_pos / R_saturn),
                'Cassini RP Distance: {:.2f} $R_S$'.format(pos[2]/R_saturn),
                'Cassini Speed: {:.1f} km/s'.format(v_cas_mag),
                'Cassini Inclination: {:.1f}\N{DEGREE SIGN}'.format(i_cas*180/pi),
                'CDA RAM Angle: {:.2f}\N{DEGREE SIGN}'.format(ram_sep),
                'CDA RP Angle: {:.2f}\N{DEGREE SIGN}'.format(ring_sep),
                'Impact Speed: {:.1f} to {:.1f} km/s'.format(v_ram_min, v_ram_max),
                'Grain Speed: {:.1f} to {:.1f} km/s'.format(v_p_mag.min(), v_p_mag.max()),
                'Grain Inclination:{:.1f} to {:.1f}\N{DEGREE SIGN}'.format(i_vals.min(), i_vals.max())))
            plt.text(25.5, 2.9, textstr, fontsize=10, va='top', ha='left', bbox=props2)
            plt.subplots_adjust(right=0.65)

            plt.savefig(home_path + '/figs/det_{0}_a_vs_e_hist_{1}.pdf'.format(sclk[det_sclk], cond_filetag))


            # HISTOGRAMS (W/ PROBABILITY WEIGHTED BINS) FOR ORBIT ORIENTATION (i, w, and lan) VS IMPACT SPEED
            # probablility weighting is set by 'weights', density=True normalizes the histograms (squares' P's sum to 1).

            fig9, axs9 = plt.subplots(1, 3, figsize=(10.75, 4.0), sharey=True, constrained_layout=True)
            x = i_vals, w_vals, lan_vals
            y = v_imp_vals
            text_x = 0.975
            text_y = 0.085
            props1 = dict(boxstyle='round', facecolor='wheat', alpha=0.6)        # create textbox style
            # place textbox with number of samples
            x_bins = 24 # 7.5 deg bins for i, 15 degree bins for w and lan (180/24 and 360/24 respectively)
            y_bins = len(v_sim_mag) # 0.5 km/s bins

            fig9.suptitle(r"Orbit Orientation vs. Impact Speed Histogram" "\n" r"{0}, SCLK: {1}".format(utc, sclk[det_sclk]),
                          size=10, weight='bold')
            axs9[0].set_ylabel('Impact Speed [km/s]')
            axs9[0].yaxis.set_major_locator(ticker.MultipleLocator(2.5))

            axs9[0].set_xlabel('Inclination, i [\N{DEGREE SIGN}]')
            axs9[0].text(text_x, text_y, n_sim, transform=axs9[0].transAxes, fontsize=8, va='top', ha='right', bbox=props1)
            # axs9[0].xaxis.set_major_locator(ticker.MultipleLocator(30))
            count, x_edges_i, y_edges_i, hist_scale_i = axs9[0].hist2d(x[0], y, bins=[x_bins, y_bins], range=((0, 180),(min(y),max(y))), density=True, weights=P_vals, cmap=plt.cm.jet, rasterized=True)
            axs9[1].set_xlabel('Arg. of Perikrone, $\omega$ [\N{DEGREE SIGN}]')
            axs9[1].text(text_x, text_y, n_sim, transform=axs9[1].transAxes, fontsize=8, va='top', ha='right', bbox=props1)
            # axs9[1].xaxis.set_major_locator(ticker.MultipleLocator(60))
            count, x_edges_w, y_edges_w, hist_scale_w = axs9[1].hist2d(x[1], y, bins=[x_bins, y_bins], range=((0, 360),(min(y),max(y))), density=True, weights=P_vals, cmap=plt.cm.jet, rasterized=True)
            # plt.colorbar(z1_plot, ax=ax1)
            axs9[2].set_xlabel('Long. of Asc. Node, $\Omega$ [\N{DEGREE SIGN}]')
            axs9[2].text(text_x, text_y, n_sim, transform=axs9[2].transAxes, fontsize=8, va='top', ha='right', bbox=props1)
            # axs9[2].xaxis.set_major_locator(ticker.MultipleLocator(60))
            count, x_edges_lan, y_edges_lan, hist_scale_lan = axs9[2].hist2d(x[2], y, bins=[x_bins, y_bins], range=((0, 360),(min(y),max(y))), density=True, weights=P_vals, cmap=plt.cm.jet, rasterized=True)

            for ax in axs9.flat:
                ax.tick_params(labelsize=8)
                ax.set_xticklabels(ax.get_xticks(), rotation=315)
                ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))  # No decimal places
                ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
                # ax.yaxis.set_major_locator(ticker.MultipleLocator(2.5))

            # make colorbars and colorbar settings
            cb_i = plt.colorbar(hist_scale_i, ax=axs9[0], orientation='horizontal')
            cb_i.ax.tick_params(rotation=0, labelsize=8)
            cb_i.ax.locator_params(nbins=6)
            cb_w = plt.colorbar(hist_scale_w, ax=axs9[1], orientation='horizontal')
            cb_w.ax.tick_params(rotation=0, labelsize=8)
            cb_w.ax.locator_params(nbins=6)
            cb_lan = plt.colorbar(hist_scale_lan, ax=axs9[2], orientation='horizontal')
            cb_lan.ax.tick_params(rotation=0, labelsize=8)
            cb_lan.ax.locator_params(nbins=6)

            plt.savefig(home_path + '/figs/det_{0}_or_vs_vimp_hist_{1}.pdf'.format(sclk[det_sclk], cond_filetag))


            fig10, axs10 = plt.subplots(1, 3, figsize=(10.75, 4.0), sharey=True, constrained_layout=True)
            x = i_vals, w_vals, lan_vals
            y = e_vals
            text_x = 0.975
            text_y = 0.085
            props1 = dict(boxstyle='round', facecolor='wheat', alpha=0.6)        # create textbox style
            # place textbox with number of samples
            # x_bins = 24 # 7.5 deg bins for i, 15 degree bins for w and lan (180/24 and 360/24 respectively)
            # y_bins = 10 # 0.5 km/s bins
            if max(y) >=10:
                x_bins, y_bins = 24, 24
            elif 6 <= max(y) < 10:
                x_bins, y_bins = 20, 20
            else:
                x_bins, y_bins = 15, 15
            fig10.suptitle(r"Orbit Orientation vs. Orbit Eccentricity Histogram" "\n" r"{0}, SCLK: {1}".format(utc, sclk[det_sclk]),
                          size=10, weight='bold')
            axs10[0].set_ylabel('Eccentricity')
            axs10[0].set_xlabel('Inclination, i [\N{DEGREE SIGN}]')
            axs10[0].text(text_x, text_y, n_sim, transform=axs10[0].transAxes, fontsize=8, va='top', ha='right', bbox=props1)
            # axs10[0].xaxis.set_major_locator(ticker.MultipleLocator(30))
            count, x_edges_i, y_edges_i, hist_scale_i = axs10[0].hist2d(x[0], y, bins=[x_bins, y_bins], range=((0, 180),(0,np.round(max(y)))), density=True, weights=P_vals, cmap=plt.cm.jet, rasterized=True)
            axs10[1].set_xlabel('Arg. of Perikrone, $\omega$ [\N{DEGREE SIGN}]')
            axs10[1].text(text_x, text_y, n_sim, transform=axs10[1].transAxes, fontsize=8, va='top', ha='right', bbox=props1)
            # axs10[1].xaxis.set_major_locator(ticker.MultipleLocator(60))
            count, x_edges_w, y_edges_w, hist_scale_w = axs10[1].hist2d(x[1], y, bins=[x_bins, y_bins], range=((0, 360),(0,np.round(max(y)))), density=True, weights=P_vals, cmap=plt.cm.jet, rasterized=True)
            # plt.colorbar(z1_plot, ax=ax1)
            axs10[2].set_xlabel('Long. of Asc. Node, $\Omega$ [\N{DEGREE SIGN}]')
            axs10[2].text(text_x, text_y, n_sim, transform=axs10[2].transAxes, fontsize=8, va='top', ha='right', bbox=props1)
            # axs10[2].xaxis.set_major_locator(ticker.MultipleLocator(60))
            count, x_edges_lan, y_edges_lan, hist_scale_lan = axs10[2].hist2d(x[2], y, bins=[x_bins, y_bins], range=((0, 360),(0,np.round(max(y)))), density=True, weights=P_vals, cmap=plt.cm.jet, rasterized=True)

            for ax in axs10.flat:
                ax.tick_params(labelsize=8)
                ax.set_xticklabels(ax.get_xticks(), rotation=315)
                ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))  # No decimal places
                ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
                ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))

            # For individual subplot colorbars and colorbar settings
            cb_i = plt.colorbar(hist_scale_i, ax=axs10[0], orientation='horizontal')
            cb_i.ax.tick_params(rotation=0, labelsize=8)
            cb_i.ax.locator_params(nbins=6)
            cb_w = plt.colorbar(hist_scale_w, ax=axs10[1], orientation='horizontal')
            cb_w.ax.tick_params(rotation=0, labelsize=8)
            cb_w.ax.locator_params(nbins=6)
            cb_lan = plt.colorbar(hist_scale_lan, ax=axs10[2], orientation='horizontal')
            cb_lan.ax.tick_params(rotation=0, labelsize=8)
            cb_lan.ax.locator_params(nbins=5)

            plt.savefig(home_path + '/figs/det_{0}_or_vs_e_hist_{1}.pdf'.format(sclk[det_sclk], cond_filetag))

            # textstr = '\n'.join((
            #     r'# samples = {:.0f}'.format(len(v_sim_mag)),
            #     r'FWHM = {:.2f}'.format(FWHM),
            #     r'$\mu$ = {:.2f} km/s'.format(v_ram_mean),
            #     r'$\sigma$ = {:.2f}'.format(stdev)))

            #
            #  2D SCATTER OF ORBITAL SHAPE
            # Plot the orbital shape elements with a color scale for relative probability
            fig5, axs5 = plt.subplots(1, 2, figsize=(7.5, 3), sharey=True, constrained_layout=True)
            x = a_vals, i_vals
            y = e_vals
            text_x = 0.975
            text_y = 0.085
            color = P_vals
            norm = plt.Normalize(P_vals.min(), P_vals.max())
            fig5.suptitle(r"Orbital Shape and Relative Probability" "\n" "{0}" "\n" r"{1}, SCLK: {2}".format(cond_plottag, utc, sclk[det_sclk]),
                          size=10, weight='bold')
            axs5[0].set_ylabel('Eccentricity, e')
            axs5[0].set_ylim(0, 3)
            axs5[0].set_xlabel('Semi-major Axis, a [$R_S$]')
            axs5[0].set_xlim(-15, 15)
            axs5[0].xaxis.set_major_locator(ticker.MultipleLocator(2))
            axs5[0].text(text_x, text_y, n_sim, transform=axs5[0].transAxes, fontsize=8, va='top', ha='right', bbox=props1)
            axs5[1].set_xlabel('Inclination, i [\N{DEGREE SIGN}]')
            axs5[1].set_xlim(0, 90)
            axs5[1].xaxis.set_major_locator(ticker.MultipleLocator(10))
            axs5[1].text(text_x, text_y, n_sim, transform=axs5[1].transAxes, fontsize=8, va='top', ha='right', bbox=props1)
            axs5[1].scatter(x[1], y, s=10, c=color, cmap=plt.jet(), alpha=0.5, rasterized=True, norm=norm)
            scatter2d_scale = axs5[0].scatter(x[0], y, s=10, c=color, cmap=plt.jet(), alpha=0.5, rasterized=True, norm=norm)

            fig5.colorbar(scatter2d_scale, ax=axs5[:]).set_label('Relative Probability', rotation=270, labelpad=15)

            plt.savefig(home_path + '/figs/det_{0}_ai_vs_e_scatter_{1}.pdf'.format(sclk[det_sclk], cond_filetag))

            #
            #  2D SCATTER OF GRAIN SPEED, SEMI-MAJOR AXIS, AND ECCENTRICITY VS. IMPACT SPEED
            # Plot the orbital shape elements with a color scale for relative probability
            # fig8, axs8 = plt.subplots(1, 3, figsize=(10.75, 3), sharey=True, constrained_layout=True)
            # x = v_p_vals, a_vals, e_vals
            # y = v_imp_vals
            # text_x = 0.975
            # text_y = 0.085
            # color = P_vals
            #
            # fig8.suptitle(
            #     r"Orbital Speed, Shape, and Relative Probability" "\n" r"{0}, SCLK: {1}".format(utc, sclk[det_sclk]),
            #     size=10, weight='bold')
            #
            # axs8[0].set_ylabel('Impact Speed [km/s]')
            # axs8[0].set_ylim(0, 30)
            # axs8[0].set_xlabel('Grain Speed [km/s]')
            # axs8[0].set_xlim(0, 20)
            # axs8[0].xaxis.set_major_locator(ticker.MultipleLocator(2))
            # axs8[0].text(text_x, text_y, n_sim, transform=axs8[0].transAxes, fontsize=8, va='top', ha='right',
            #              bbox=props1)
            # axs8[0].scatter(x[0], v_imp_vals, s=10, c=color, cmap=plt.jet(), alpha=0.5, rasterized=True, norm=norm)
            #
            # axs8[1].set_xlabel('Semi-major Axis, a [$R_S$]')
            # axs8[1].set_xlim(-15, 15)
            # axs8[1].xaxis.set_major_locator(ticker.MultipleLocator(10))
            # axs8[1].text(text_x, text_y, n_sim, transform=axs8[1].transAxes, fontsize=8, va='top', ha='right',
            #              bbox=props1)
            # axs8[1].scatter(x[1], y, s=10, c=color, cmap=plt.jet(), alpha=0.5, rasterized=True, norm=norm)
            #
            # axs8[2].set_ylabel('Impact Speed [km/s]')
            # axs8[2].set_xlabel('Eccentricity')
            # axs8[2].set_xlim(0, 3)
            # axs8[2].xaxis.set_major_locator(ticker.MultipleLocator(2))
            # axs8[2].text(text_x, text_y, n_sim, transform=axs8[2].transAxes, fontsize=8, va='top', ha='right',
            #              bbox=props1)
            # im = axs8[2].scatter(x[2], y, s=10, c=color, cmap=plt.jet(), alpha=0.5, rasterized=True, norm=norm)
            #
            # # scatter2d_scale = axs8[0].scatter(v_p_vals, v_imp_vals, s=10, c=P_vals, cmap=plt.jet(), alpha=0.5, rasterized=True)
            #
            # fig8.colorbar(im, ax=axs8[2], aspect=50).set_label('Relative Probability', rotation=270, labelpad=15)


            # these are matplotlib.patch.Patch properties
            # props1 = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            # place a text box in upper left in axes coords


            #
            # SIX PLOT SCATTER - ORBITS BINNED BY INCLINATION 10 or 5 DEG INCREMENTS
            #
            data = [data_p_i10, data_p_i20, data_p_i30, data_p_i40, data_p_i50, data_p_i60, data_p_i70, data_p_i80, data_p_i90]
            # data = [data_p_i10, data_p_i20, data_p_i30, data_p_i40, data_p_i45, data_p_i50f, data_p_i55, data_p_i60f, data_p_i65, data_p_i70f]
            color =     [ data_p_i10[:,10]/sum(data_p[:,10]), data_p_i20[:,10]/sum(data_p[:,10]), data_p_i30[:,10]/sum(data_p[:,10]),
                        data_p_i40[:,10]/sum(data_p[:,10]), data_p_i50[:,10]/sum(data_p[:,10]), data_p_i60[:,10]/sum(data_p[:,10]),
                        data_p_i70[:,10]/sum(data_p[:,10]), data_p_i80[:,10]/sum(data_p[:,10]), data_p_i90[:,10]/sum(data_p[:,10])]
            point_count = [r'n = {:.0f}'.format(len(data[0])), r'n = {:.0f}'.format(len(data[1])), r'n = {:.0f}'.format(len(data[2])),
                        r'n = {:.0f}'.format(len(data[3])), r'n = {:.0f}'.format(len(data[4])), r'n = {:.0f}'.format(len(data[5])),
                        r'n = {:.0f}'.format(len(data[6])), r'n = {:.0f}'.format(len(data[7])), r'n = {:.0f}'.format(len(data[8]))]

            fig6, axs6 = plt.subplots(3, 3, figsize=(7.5,9), sharey=True, constrained_layout=True)
            fig6.suptitle(r"Orbital Shape and Relative Probability by Inclination" "\n" "{0}" "\n" r"{1}, SCLK: {2}".format(cond_plottag, utc, sclk[det_sclk]),
                          size=10, weight='bold')

            for ax in axs6.flat:
                ax.set_xlabel('Semi-major Axis, a [$R_S$]', fontsize=8)
                # ax.set_ylabel('Eccentricity, e', fontsize=8)
                ax.tick_params(labelsize=8)

            text_x = 0.97
            text_y = 0.06
            xmin, xmax = -20, 20
            ymin, ymax = 0, 3


            im1 = axs6[0, 0].scatter(data_p_i10[:,5], data_p_i10[:,6], s=4, c=color[0], cmap=plt.jet(), alpha=0.5, rasterized=True, norm=norm)
            axs6[0, 0].set_title("0-10\N{DEGREE SIGN}", fontsize=10)
            axs6[0, 0].set_ylabel('Eccentricity, e', fontsize=8)
            axs6[0, 0].set_xlim(xmin, xmax)
            axs6[0, 0].set_ylim(ymin, ymax)
            axs6[0, 0].text(text_x, text_y, point_count[0], transform=axs6[0,0].transAxes, fontsize=6, va='top', ha='right', bbox=props1)
            im2 = axs6[0, 1].scatter(data_p_i20[:,5], data_p_i20[:,6], s=4, c=color[1], cmap=plt.jet(), alpha=0.5, rasterized=True, norm=norm)
            axs6[0, 1].set_title("10-20\N{DEGREE SIGN}", fontsize=10)
            axs6[0, 1].set_xlim(xmin, xmax)
            axs6[0, 1].set_ylim(ymin, ymax)
            axs6[0, 1].text(text_x, text_y, point_count[1], transform=axs6[0,1].transAxes, fontsize=6, va='top', ha='right', bbox=props1)
            im3  = axs6[0, 2].scatter(data_p_i30[:,5], data_p_i30[:,6], s=4, c=color[2], cmap=plt.jet(), alpha=0.5, rasterized=True, norm=norm)
            axs6[0, 2].set_title("20-30\N{DEGREE SIGN}", fontsize=10)
            axs6[0, 2].set_xlim(xmin, xmax)
            axs6[0, 2].set_ylim(ymin, ymax)
            axs6[0, 2].text(text_x, text_y, point_count[2], transform=axs6[0,2].transAxes, fontsize=6, va='top', ha='right', bbox=props1)
            im4 = axs6[1, 0].scatter(data_p_i40[:,5], data_p_i40[:,6], s=4, c=color[3], cmap=plt.jet(), alpha=0.5, rasterized=True, norm=norm)
            axs6[1, 0].set_title("30-40\N{DEGREE SIGN}", fontsize=10)
            axs6[1, 0].set_ylabel('Eccentricity, e', fontsize=8)
            axs6[1, 0].set_xlim(xmin, xmax)
            axs6[1, 0].set_ylim(ymin, ymax)
            axs6[1, 0].text(text_x, text_y, point_count[3], transform=axs6[1,0].transAxes, fontsize=6, va='top', ha='right', bbox=props1)
            im5 = axs6[1, 1].scatter(data_p_i50[:,5], data_p_i50[:,6], s=4, c=color[4], cmap=plt.jet(), alpha=0.5, rasterized=True, norm=norm)
            axs6[1, 1].set_title("40-50\N{DEGREE SIGN}", fontsize=10)
            axs6[1, 1].set_xlim(xmin, xmax)
            axs6[1, 1].set_ylim(ymin, ymax)
            axs6[1, 1].text(text_x, text_y, point_count[4], transform=axs6[1,1].transAxes, fontsize=6, va='top', ha='right', bbox=props1)
            im6 = axs6[1, 2].scatter(data_p_i60[:,5], data_p_i60[:,6] , s=4, c=color[5], cmap=plt.jet(), alpha=0.5, rasterized=True, norm=norm)
            axs6[1, 2].set_title("50-60\N{DEGREE SIGN}", fontsize=10)
            axs6[1, 2].set_xlim(xmin, xmax)
            axs6[1, 2].set_ylim(ymin, ymax)
            axs6[1, 2].text(text_x, text_y, point_count[5], transform=axs6[1,2].transAxes, fontsize=6, va='top', ha='right', bbox=props1)
            im7 = axs6[2, 0].scatter(data_p_i70[:,5], data_p_i70[:,6], s=4, c=color[6], cmap=plt.jet(), alpha=0.5, rasterized=True, norm=norm)
            axs6[2, 0].set_title("60-70\N{DEGREE SIGN}", fontsize=10)
            axs6[2, 0].set_ylabel('Eccentricity, e', fontsize=8)
            axs6[2, 0].set_xlim(xmin, xmax)
            axs6[2, 0].set_ylim(ymin, ymax)
            axs6[2, 0].text(text_x, text_y, point_count[6], transform=axs6[2,0].transAxes, fontsize=6, va='top', ha='right', bbox=props1)
            im8 = axs6[2, 1].scatter(data_p_i80[:,5], data_p_i80[:,6], s=4, c=color[7], cmap=plt.jet(), alpha=0.5, rasterized=True, norm=norm)
            axs6[2, 1].set_title("70-80\N{DEGREE SIGN}", fontsize=10)
            axs6[2, 1].set_xlim(xmin, xmax)
            axs6[2, 1].set_ylim(ymin, ymax)
            axs6[2, 1].text(text_x, text_y, point_count[7], transform=axs6[2,1].transAxes, fontsize=6, va='top', ha='right', bbox=props1)
            im9 = axs6[2, 2].scatter(data_p_i90[:,5], data_p_i90[:,6] , s=4, c=color[8], cmap=plt.jet(), alpha=0.5, rasterized=True, norm=norm)
            axs6[2, 2].set_title("80-90\N{DEGREE SIGN}", fontsize=10)
            axs6[2, 2].set_xlim(xmin, xmax)
            axs6[2, 2].set_ylim(ymin, ymax)
            axs6[2, 2].text(text_x, text_y, point_count[8], transform=axs6[2,2].transAxes, fontsize=6, va='top', ha='right', bbox=props1)

            cbar = fig6.colorbar(scatter2d_scale, ax=axs6[:, 2], aspect=50).set_label('Relative Probability', rotation=270, labelpad=15)

            plt.savefig(home_path + '/figs/det_{0}_aei_scatter_{1}.pdf'.format(sclk[det_sclk], cond_filetag))

            # # RUN CHECKS ON ORBIT MODEL DATA
            # print("# Velocities modeled:", len(v_sim_mag), "\n",
            # "# theta angles modeled:", len(theta), "\n",
            # "# phi angles modeled:", len(phi), "\n",
            # '# Total # orbit models:', len(data_p), "\n")

            # print("# Circular or near-circular orbits:", n_cir, "of", len(e_vals), "\n",
            # "# Elliptical orbits:", n_ell, "of", len(e_vals), "\n",
            # "# Parabolic or hyperbolic orbtis:", n_phy, "of", len(e_vals))
            #

            # examine the solid angle weight factor according to theta angle.
            # for i in range(len(theta)):
            #     print("theta:", theta[i]*180/pi, "SA wf:", SA_wf[i])

            #
            # CHECK IMPACT VECTORS AROUND BORESIGHT
            #
            # ax = plt.figure().add_subplot(projection='3d')
            # ax.quiver(0,0,0, v_ram_check[:,0], v_ram_check[:,1], v_ram_check[:,2], arrow_length_ratio=0.05, linewidth=0.25, color='black')
            # ax.set_xlim([-0.75, 0.75])
            # ax.set_ylim([-0.75, 0.75])
            # ax.set_zlim([0, 1])

            #
            # # CHECK IMPACT VECTORS AROUND BORESIGHT 2D
            #
            # fig, ax = plt.subplots()
            # o_val = np.zeros(len(v_ram_check[:,0]))
            # origin = np.array([o_val, o_val])  # origin point
            #
            # # plt.quiver(*origin, V[:, 0], V[:, 1], color=['r', 'b', 'g'], scale=21)
            #
            # ax.quiver(*origin, v_ram_check[:,0], v_ram_check[:,2], scale=2)
            # ax.set_xlim([-0.75, 0.75])
            # ax.set_ylim([0, 1])

            det_sum[det_sclk] = [sclk[det_sclk],                            # 0 detection SCLK time
                              np.round((R_pos/R_saturn), decimals=2),       # 1 Cassini distance [R_s]
                              np.round((pos[2]/R_saturn), decimals=2),      # 2 Cassini RP distance [R_s]
                              np.round(v_cas_mag, decimals=1),              # 3 Cassini speed [km/s]
                              np.round((i_cas*180/pi), decimals=1),         # 4 Cassini inclination [deg]
                              np.round(ram_sep, decimals=2),                # 5 CDA RAM angle [deg]
                              np.round(ring_sep, decimals=2),               # 6 CDA RP angle [deg]
                              np.round(v_ram_min, decimals=1),              # 7 V_imp min [km/s]
                              np.round(v_ram_max, decimals=1),              # 8 V_imp max
                              np.round(v_p_mag.min(), decimals=1),          # 9 Grain speed min [km/s]
                              np.round(v_p_mag.max(), decimals=1),          # 10 grain speed max
                              np.round(i_vals.min(), decimals =1),          # 11 min grain inclination [deg]
                              np.round(i_vals.max(), decimals=1),           # 12 max grain inclination
                              np.round(np.mean(i_vals),decimals=1),          # 13 i average [deg]
                              np.round(w_vals.min(), decimals =1),          # 14 min omega [deg]
                              np.round(w_vals.max(), decimals=1),           # 15 max omega [deg]
                              np.round(np.mean(w_vals),decimals=1),          # 16 omega average [deg]
                              np.round(lan_vals.min(), decimals =1),        # 17 min LAN [deg]
                              np.round(lan_vals.max(), decimals=1),         # 18 max LAN [deg]
                              np.round(np.mean(lan_vals),decimals=1),        # 19 LAN average [deg]
                              len(i_vals)                                   # 20 total sample # of modeled orbits
                              ]
            det_data = np.append(det_data, det_sum[det_sclk].reshape(1,n_var_detdata), axis=0)

        # # CREATE DATAFRAME AND EXPORT DETECTION SUMMARY DATA TO CSV AS NEEDED.
        df_det = pd.DataFrame({"det SCLK": det_data[:,0],
                                "Cas R [R_s]": det_data[:,1],
                                "Cas RP dist [R_s]": det_data[:,2],
                                "Cas Speed [km/s]": det_data[:,3],
                                "Cassini i [deg]": det_data[:,4],
                                "CDA RAM Angle": det_data[:,5],
                                "CDA RP Angle": det_data[:,6],
                                "V_imp min [km/s]": det_data[:,7],
                                "V_imp max [km/s]": det_data[:,8],
                                "Grain v min [km/s]": det_data[:,9],
                                "Grain v max [km/s]": det_data[:,10],
                                "Grain i min [deg]": det_data[:,11],
                                "Grain i max [deg]": det_data[:,12],
                                "Grain i avg [deg]": det_data[:,13],
                                "Grain w min [deg]": det_data[:,14],
                                "Grain w max [deg]": det_data[:,15],
                                "Grain w avg [deg]": det_data[:,16],
                                "Grain LAN min [deg]": det_data[:,17],
                                "Grain LAN max [deg]": det_data[:,18],
                                "Grain LAN avg [deg]": det_data[:,19], 
                                "n samples": det_data[:,20]
                                })
        df_det.to_csv('output/{}_{}_det_summary.csv'.format(filename, cond_filetag), index=False)  # comment here to suppress



    # Define plot variables for all revolutions/whole read file (cumulative data of detections within)
    if len(data_rev) > 0:
        n_sim_rev = r'n = {:.0f}'.format(len(data_rev))    # number of modeled orbits string
        v_imp_vals_rev = data_rev[:,0]
        a_vals_rev = data_rev[:,5]
        e_vals_rev = data_rev[:,6]
        i_vals_rev = data_rev[:,7]
        w_vals_rev = data_rev[:,8]
        wbar_ls_vals_rev = data_rev[:,13]
        lan_vals_rev = data_rev[:,14]

        P_vals_rev = data_rev[:,19]

        gamma = 0.5


        #
        # CUMULATIVE HISTOGRAM OF ORBITAL SHAPE (a VS. e) FOR ALL DETECTIONS IN READ FILE
        #
        fig7, ax7 = plt.subplots(1, figsize=(8.5, 4))
        fig7.suptitle(
            r"Cumulative Orbit Shape (a vs. e) Histogram, " "{0}" "\n" "{1}"
                .format(cas_orbit, cond_plottag),
            size=10, weight='bold')
        x = a_vals_rev
        y = e_vals_rev
        ax7.set_xlabel('Semi-major Axis, a [$R_S$]')
        ax7.set_ylabel('Eccentricity, e')
        # text coordinates
        text_x = 0.985
        text_y = 0.05
        props1 = dict(boxstyle='round', facecolor='wheat', alpha=0.6)  # create textbox style
        # place textbox with number of samples
        ax7.text(text_x, text_y, n_sim_rev, transform=ax7.transAxes, fontsize=8, va='top', ha='right', bbox=props1)
        x_bins = np.linspace(-15, 15, 21, endpoint=True) # (-15, 15, 21) for standard, (-5, 5, 21) for zoomed look at unconstrained dataset
        y_bins = np.linspace(0, 3, 16, endpoint=True)
        count, x_edges, y_edges, hist_scale = ax7.hist2d(x, y, bins=[x_bins, y_bins], density=False, weights=P_vals_rev,
                                                         cmap=plt.cm.jet, norm=mcolors.PowerNorm(gamma), rasterized=True)
        fig7.colorbar(hist_scale).set_label('Probability-weighted frequency', rotation=270, labelpad=18)
        props2 = dict(boxstyle='square', facecolor='white')
        textstr = '\n'.join((
            'Cassini Distance: {:.2f}–{:.2f} $R_S$'.format(data_rev[:,18].min(), data_rev[:,18].max()),
            'CDA RAM Angle: {:.2f}–{:.2f}\N{DEGREE SIGN}'.format(data_rev[:,16].min(), data_rev[:,16].max()),
            'CDA RP Angle: {:.2f}–{:.2f}\N{DEGREE SIGN}'.format(data_rev[:,17].min(), data_rev[:,17].max()),
            'Grain Speed: {:.1f}–{:.1f} km/s'.format(data_rev[:,3].min(), data_rev[:,3].max()),
            'Grain Inclination:{:.1f}–{:.1f}\N{DEGREE SIGN}'.format(i_vals_rev.min(), i_vals_rev.max()),
            ))
        plt.text(25.5, 2.9, textstr, fontsize=10, va='top', ha='left', bbox=props2)
        plt.subplots_adjust(right=0.65)

        plt.savefig(home_path + '/figs/{0}_hist-{1}-ae.pdf'.format(cas_orbit, cond_filetag))

    #
        # CUMULATIVE HISTOGRAMS (W/ PROBABILITY WEIGHED BINS) FOR ORBIT ORIENTATION (i, w, and lan) VS IMPACT SPEED
        # FOR ALL DETECTIONS IN READ FILE
        # probablility weighting is set by 'weights', density=True normalizes the histograms (squares' P's sum to 1).
        #

        fig11, axs11 = plt.subplots(1, 3, figsize=(10.75, 4.5), sharey=True, constrained_layout=True)
        x = i_vals_rev, w_vals_rev, lan_vals_rev
        y = v_imp_vals_rev
        text_x = 0.975      # x position of textbox
        text_y = 0.085      # y position of texbox
        props1 = dict(boxstyle='round', facecolor='wheat', alpha=0.6)  # create textbox style
        # place textbox with number of samples
        x_bins = 24  # 7.5 deg bins for i, 15 degree bins for w and lan (180/24 and 360/24 respectively)
        y_bins = 20 #len(v_sim_mag)  # 0.5 km/s bins   # 20 for bulk dataset (10-50km/s), 15 for constr (10-25 km/s), 14 for unconstr (15-50 km/s)

        fig11.suptitle(r"Cumulative Orbit Orientation vs. Impact Speed Histogram, " "{0}" "\n" 
                       "{1}".format(cas_orbit, cond_plottag), size=10, weight='bold')
        axs11[0].set_ylabel('Impact Speed [km/s]')
        axs11[0].set_xlabel('Inclination, i [\N{DEGREE SIGN}]')
        axs11[0].text(text_x, text_y, n_sim_rev, transform=axs11[0].transAxes, fontsize=8, va='top', ha='right', bbox=props1)
        # axs11[0].xaxis.set_major_locator(ticker.MultipleLocator(30))

        count, x_edges_i, y_edges_i, hist_scale_i = axs11[0].hist2d(x[0], y, bins=[x_bins, y_bins],
                                                                     range=((0, 180), (min(y), max(y))), density=False,
                                                                     weights=P_vals_rev, cmap=plt.cm.jet, norm=mcolors.PowerNorm(gamma), rasterized=True)
        axs11[1].set_xlabel('Arg. of Perikrone, $\omega$ [\N{DEGREE SIGN}]')
        axs11[1].text(text_x, text_y, n_sim_rev, transform=axs11[1].transAxes, fontsize=8, va='top', ha='right', bbox=props1)

        count, x_edges_w, y_edges_w, hist_scale_w = axs11[1].hist2d(x[1], y, bins=[x_bins, y_bins],
                                                                     range=((0, 360), (min(y), max(y))), density=False,
                                                                     weights=P_vals_rev, cmap=plt.cm.jet, norm=mcolors.PowerNorm(gamma), rasterized=True)
        axs11[2].set_xlabel('Long. of Asc. Node, $\Omega$ [\N{DEGREE SIGN}]')
        axs11[2].text(text_x, text_y, n_sim_rev, transform=axs11[2].transAxes, fontsize=8, va='top', ha='right', bbox=props1)
        count, x_edges_lan, y_edges_lan, hist_scale_lan = axs11[2].hist2d(x[2], y, bins=[x_bins, y_bins],
                                                                                 range=((0, 360), (min(y), max(y))),
                                                                                 density=False, weights=P_vals_rev,
                                                                                 cmap=plt.cm.jet, norm=mcolors.PowerNorm(gamma), rasterized=True)

        for ax in axs11.flat:
            ax.tick_params(labelsize=8)
            ax.set_xticklabels(ax.get_xticks(), rotation=315)
            ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))  # No decimal places
            ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
            # ax.yaxis.set_major_locator(ticker.MultipleLocator(2.5))

        # make colorbars and colorbar settings
        cb_i = plt.colorbar(hist_scale_i, ax=axs11[0], orientation='horizontal')
        cb_i.ax.tick_params(rotation=315, labelsize=8)
        # cb_i.ax.locator_params(nbins=6)
        cb_w = plt.colorbar(hist_scale_w, ax=axs11[1], orientation='horizontal')
        cb_w.set_label('Probability-weighted Frequency', labelpad=10)
        cb_w.ax.tick_params(rotation=315, labelsize=8)
        # cb_w.ax.locator_params(nbins=6)
        cb_lan = plt.colorbar(hist_scale_lan, ax=axs11[2], orientation='horizontal')
        cb_lan.ax.tick_params(rotation=315, labelsize=8)
        # cb_lan.ax.locator_params(nbins=6)

        plt.savefig(home_path + '/figs/{0}_hist-{1}-orvsvimp.pdf'.format(cas_orbit, cond_filetag))

    # single working colorbar for all plots, attributed only to one designated plot scale, i.e. hist_scale_lan
        # could set this colorbar to have custom ticks that go from 0 to 1, despite relative probabilities being different
        # fig11.colorbar(hist_scale_lan, ).set_label('Probability-weighted frequency', rotation=270, labelpad=18)


        #
        # CUMULATIVE HISTOGRAMS (W/ PROBABILITY WEIGHED BINS) FOR ORBIT ORIENTATION (i, w, and lan) VS ORBIT SHAPE (e)
        # FOR ALL DETECTIONS IN READ FILE
        # probablility weighting is set by 'weights', density=True normalizes the histograms (squares' P's sum to 1).
        # Manually limit displayed e values to 6 for bulk dataset.

        fig12, axs12 = plt.subplots(1, 3, figsize=(10.75, 4.5), sharey=True, constrained_layout=True)
        x = i_vals_rev, w_vals_rev, lan_vals_rev
        y = e_vals_rev
        text_x = 0.975
        text_y = 0.085
        props1 = dict(boxstyle='round', facecolor='wheat', alpha=0.6)  # create textbox style
        # place textbox with number of samples
        x_bins = 24  # 7.5 deg bins for i, 15 degree bins for w and lan (180/24 and 360/24 respectively)
        y_bins = 15  # 0.2 e bins for {e|0,3}
        y_max = 3 #max(e_vals_rev)    # set max e-value displayed to 3 for visibility and clarity

        fig12.suptitle(
            r"Cumulative Orbit Orientation vs. Orbit Eccentricity Histogram, " "{}" "\n" 
            "{}".format(cas_orbit, cond_plottag), size=10, weight='bold')
        axs12[0].set_ylabel('Eccentricity')
        axs12[0].set_xlabel('Inclination, i [\N{DEGREE SIGN}]')
        axs12[0].text(text_x, text_y, n_sim_rev, transform=axs12[0].transAxes, fontsize=8, va='top', ha='right', bbox=props1)

        count_i, x_edges_i, y_edges_i, hist_scale_i = axs12[0].hist2d(x[0], y, bins=[x_bins, y_bins],
                                                                      range=((0, 180), (min(y), y_max)),
                                                                      density=False, weights=P_vals_rev, cmap=plt.cm.jet,
                                                                      norm=mcolors.PowerNorm(gamma), rasterized=True)
        axs12[1].set_xlabel('Arg. of Perikrone, $\omega}$ [\N{DEGREE SIGN}]')
        # axs12[1].set_xlabel(r'Long. of Perikrone, $\bar{\omega}\,$' u"[\N{DEGREE SIGN}]") # for w_bar
        axs12[1].text(text_x, text_y, n_sim_rev, transform=axs12[1].transAxes, fontsize=8, va='top', ha='right', bbox=props1)

        count_w, x_edges_w, y_edges_w, hist_scale_w = axs12[1].hist2d(x[1], y, bins=[x_bins, y_bins],
                                                                      range=((0, 360), (min(y), y_max)),
                                                                      density=False, weights=P_vals_rev, cmap=plt.cm.jet,
                                                                      norm=mcolors.PowerNorm(gamma), rasterized=True)
        axs12[2].set_xlabel('Long. of Asc. Node, $\Omega$ [\N{DEGREE SIGN}]')
        axs12[2].text(text_x, text_y, n_sim_rev, transform=axs12[2].transAxes, fontsize=8, va='top', ha='right', bbox=props1)
        count_lan, x_edges_lan, y_edges_lan, hist_scale_lan = axs12[2].hist2d(x[2], y, bins=[x_bins, y_bins],
                                                                            range=((0, 360), (min(y), y_max)),
                                                                            density=False, weights=P_vals_rev, cmap=plt.cm.jet,
                                                                            norm=mcolors.PowerNorm(gamma), rasterized=True)

        for ax in axs12.flat:
            ax.tick_params(labelsize=8)
            ax.set_xticklabels(ax.get_xticks(), rotation=315)
            ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))  # No decimal places
            ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))

        # For individual subplot colorbars and colorbar settings
        cb_i = plt.colorbar(hist_scale_i, ax=axs12[0], orientation='horizontal')
        cb_i.ax.tick_params(rotation=315, labelsize=8)
        # cb_i.ax.locator_params(nbins=6)
        cb_w = plt.colorbar(hist_scale_w, ax=axs12[1], orientation='horizontal')
        cb_w.set_label('Probability-weighted Frequency', labelpad=10)
        cb_w.ax.tick_params(rotation=315, labelsize=8)
        # cb_w.ax.locator_params(nbins=6)
        cb_lan = plt.colorbar(hist_scale_lan, ax=axs12[2], orientation='horizontal')
        cb_lan.ax.tick_params(rotation=315, labelsize=8)
        # cb_lan.ax.locator_params(nbins=6)

        plt.savefig(home_path + '/figs/{0}_hist-{1}-orvse.pdf'.format(cas_orbit, cond_filetag))

    # CREATE DATAFRAME AND EXPORT OUTPUT DATA TO CSV AS NEEDED.
    df = pd.DataFrame({"det SCLK": data_rev[:, 15],
                        "v_sim_mag [km/s]": np.round(data_rev[:, 0], decimals=2), "theta [deg]":
                        np.round(data_rev[:, 1], decimals=1), "phi [deg]": np.round(data_rev[:, 2], decimals=2),
                       "v_p_mag [km/s]": np.round(data_rev[:, 3], decimals=2),
                       "a [km]": np.round(data_rev[:, 4]), "a [R_s]": np.round(data_rev[:, 5], decimals=2),
                       "e": np.round(data_rev[:, 6], decimals=2),
                       "i [deg]": np.round(data_rev[:, 7], decimals=2), "w (omega)": np.round(data_rev[:, 8], decimals=2),
                       "lan": np.round(data_rev[:, 9], decimals=2),
                       "P(orbit)": data_rev[:, 10], "P(theta)": data_rev[:, 11], "P(v_imp)": data_rev[:, 12],
                       "w_bar_Ls": np.round(data_rev[:, 13], decimals=2), "lan_Ls": np.round(data_rev[:, 14], decimals=2),
                       "RAM angle": data_rev[:, 16], "RP angle": data_rev[:, 17], "SC pos": data_rev[:, 18], "P_norm": data_rev[:, 19],
                       "RP Dist": data_rev[:, 20]
                       })
    df.to_csv('output/{}_{}_data.csv'.format(filename, cond_filetag), index=False)  # comment here to suppress

    # Save figures as pdf: for-fig loop should be inside of for-sclk loop for individual detection pdfs, and
    # outside of the sclk loop for a cumulative pdf of all detections in the data csv.
    # Omit plt.close for cumulative pdf.
    # pdf = matplotlib.backends.backend_pdf.PdfPages('output/report_{0}_{1}.pdf'.format(cas_orbit, cond_filetag))

    # comment next four lines to suppress
    for fig in range(1, plt.gcf().number + 1):
        pdf.savefig(fig)
    pdf.close()
    plt.close('all')  # place after pdf.close for cumulative pdf

    # plt.show()     # keep off to suppress live plotting

if __name__ == '__main__':
    sims_PO()
    time_f = datetime.now()
    print("time elapsed:", time_f-time_i)


    # import sys
    # if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    #     QtGui.QApplication.instance().exec_()

# numpy.version.version
# print(type(var))
# print(np.shape(var))
# print(matplotlib.__version__)
# user_input_example = int(input("Enter the index in the list : "))