



def vector_add():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    import matplotlib.ticker as ticker

    # soa = np.array([[0, 0, 0, 1, -2, 0],
    #                 [0, 0, 0, 1, 1, 0],
    #                 [2, -1, 0, 0, 0, 0]])
    #
    # V = np.array([[1, -2, 0], [1, 2, 0], [0, 0, 1]])
    # origin = np.array([[0, 0, -1], [0, 0, 3], [0, 0, 1]])  # origin point

    v_cas = np.array([13 , 3 ,-5])
    v_imp = np.array([-12.5, 7 ,3.5])
    v3 = np.array(v_cas+v_imp)

    print('v_cas', np.linalg.norm(v_cas))
    print('v_imp', np.linalg.norm(v_imp))

    o1 = np.array([0, 0 ,0])
    o2 = np.array([0, 0 ,0])
    o3 = np.array([0, 0 ,0])

    x1, y1, z1 = o1
    u1, v1, w1 = v_cas
    x2, y2, z2 = o2
    u2, v2, w2 = v_imp
    x3, y3, z3 = o3
    u3, v3, w3 = v3
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.quiver(x1, y1, z1, u1, v1, w1, color='r', label='V_Cassini')
    plt.quiver(x2, y2, z2, u2, v2, w2, color='g', label='V_imp')
    plt.quiver(x3, y3, z3, u3, v3, w3, color='b', label='V_grain')

    ax.set_xlim([-20, 20])
    ax.set_ylim([-5, 20])
    ax.set_zlim([-10, 10])

    ax.set_xlabel('X [km/s]')
    ax.set_ylabel('Y [km/s]')
    ax.set_zlabel('Z [km/s]')
    ax.grid(which='both', color='gray', linewidth=0.25)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(5))

    h = []  # initiate handles array
    l = []  # initiate labels array
    axLine, axLabel = ax.get_legend_handles_labels()
    h.extend(axLine)
    l.extend(axLabel)
    ax.legend(h, l, bbox_to_anchor=(.85, .8), fontsize=8, loc='upper right', borderaxespad=0.)


    plt.show()








if __name__ == '__main__':
        vector_add()