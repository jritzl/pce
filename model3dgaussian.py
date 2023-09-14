
import multiprocessing

import scipy.io as sio

import chaospy as cp
import numpy as np


# mat_contents = sio.loadmat("C:\\Users\\Yunus\\Desktop\\pyy\\pat5ww.mat")

# evals = mat_contents['data']

mat_contents = sio.loadmat("C:\\Users\\Yunus\\Desktop\\pceresearch\\pyy\\patSkullthree.mat")
evals = mat_contents['data']


# skin=cp.Normal(1624,91.8)
# csf=cp.Normal(1504.5,3.5)
# cerebellum=cp.Normal(1552.5,29.0)
# gm=cp.Normal(1552.5,29.0)
# wm=cp.Normal(1552.5,29.0)
skull=cp.Normal(2813.7,337)
ventricles=cp.Normal(1504.5,3.5)
# amygdala=cp.Normal(1552.5,29.0)
distribution=cp.J(skull,ventricles)
nodes,weights = cp.generate_quadrature(2, distribution, rule="gaussian" )
print(nodes)

# sample0=[1616.987134906245, 1635.4164548717913, 1634.790636094854, 1623.777849958655, 1623.7847377383553, 1604.8295299597385, 1628.7594480385121, 1630.0409296627197, 1610.177986846923, 1624.8452797754737, 1624.2865773509554, 1636.5733551876715, 1618.380482056651, 1619.0499294833444, 1618.8534492290369, 1633.4211884683352, 1632.571531598274, 1633.730949628237, 1620.8451587423735, 1605.3771724570336, 1621.8397798183726, 1627.8539764004329, 1611.8297346763109, 1600.8623279632495, 1620.1174172430133, 1635.0938478221426, 1623.6761457290072, 1625.6531217483519, 1625.7517659677508, 1632.8306348589679]
# sample1=[1507.942474817312, 1504.4393229145908, 1500.5140144664515, 1503.3638939527902, 1503.0432611674164, 1506.887857192463, 1505.4383807851752, 1502.0717692093485, 1501.8385577749307, 1503.7246128409654, 1505.4692481768707, 1507.1872831976646, 1503.5218386884258, 1506.8647973989794, 1503.9259841899027, 1503.0549303893029, 1504.6644271674218, 1504.2368678307578, 1503.317369164009, 1505.3771940018862, 1503.5106052863232, 1503.187017083981, 1505.9159172705256, 1505.071940381596, 1504.2786515831872, 1503.6406152407988, 1500.1022916989718, 1508.6876589612311, 1504.3722857688301, 1504.1308680322584]
# sample2=[1562.8028843651337, 1545.2456864532305, 1552.3296695932531, 1563.8790142512646, 1557.70986118024, 1550.9822530380748, 1544.0808313323905, 1548.7259594532368, 1547.2799582945133, 1544.1597485326595, 1563.280412547297, 1550.2482165795063, 1551.6499985596863, 1542.943325781487, 1546.1260709880985, 1557.8276727683176, 1558.0206329852258, 1548.2630515100714, 1552.7294544294086, 1558.916732075049, 1542.5394945837625, 1547.1191383024677, 1557.4422910839737, 1540.6932433447662, 1554.0636730296596, 1559.456251479422, 1556.69347201152, 1558.0235758185524, 1549.1400092291692, 1551.64574477175]
# sample3=[1551.8166123503545, 1548.1183824612071, 1558.505996072499, 1552.0627522317864, 1554.8638327100484, 1539.4641348579612, 1562.8072493141008, 1553.175156147074, 1549.2644251413262, 1546.4983071639094, 1547.54046396888, 1550.2345538633042, 1553.113924360685, 1553.0230173305688, 1555.4417485103338, 1551.3043896729348, 1557.6479080655984, 1556.9009184997926, 1544.5968245113731, 1558.374595438147, 1553.995218806355, 1550.292610478685, 1557.1814404016452, 1554.0475710975086, 1553.486979802682, 1552.193709195387, 1551.2049044771868, 1550.4915039847433, 1551.9208439791812, 1563.388382361483]
# sample4=[1546.5166843550548, 1553.5791918846846, 1546.700172529395, 1553.8452500087835, 1539.4355010873142, 1549.9774023719679, 1556.3022139051384, 1550.3920923140433, 1558.3309004030077, 1557.7259914838112, 1563.86500443276, 1557.0518971693564, 1561.0322806417078, 1556.1257046237558, 1550.9934080736416, 1556.6393328142026, 1551.1995163110382, 1548.7222691694853, 1544.0458237294781, 1553.4603585370176, 1544.774381285934, 1557.0794861670322, 1547.5785320985929, 1558.0084361445552, 1551.861507123908, 1557.4941193104444, 1545.520876129257, 1547.897081111689, 1543.7272402188105, 1557.2850129582632]
# sample5=[2802.35920008988, 2827.082280442884, 2776.1986823020247, 2829.3030508418587, 2823.7043327301144, 2799.120504561064, 2821.1518029193608, 2808.6106432934944, 2816.09246223174, 2783.7631335244146, 2824.248193822582, 2806.060717349456, 2822.2179069365448, 2787.9500813086047, 2818.3713113972835, 2825.0848888399996, 2838.872772855364, 2801.83734940373, 2812.747893366792, 2817.5278918758836, 2788.5740515461844, 2841.0170418193034, 2829.6109856772187, 2806.323847199508, 2797.3342839651496, 2825.5041867342243, 2804.9936376524342, 2800.9496307254744, 2828.4001483160023, 2835.0244294720474]
# sample6=[1501.9426008233895, 1505.0266376371774, 1502.5599715363726, 1505.606832035867, 1503.255901610517, 1503.7351735757022, 1505.475448437414, 1502.4288640149973, 1504.5986395607633, 1503.9583249560892, 1504.814508365413, 1504.6889849368151, 1506.7151074722053, 1505.966240213008, 1507.6859405663656, 1506.8602947570062, 1504.3280524780373, 1508.7560537657614, 1503.1127458062635, 1505.1768212036009, 1505.4117000839167, 1501.6580125119333, 1501.3810640705212, 1502.6545676323174, 1506.4826112588148, 1504.9824087287866, 1506.9616550505148, 1502.2463067967844, 1504.5840161071053, 1502.4137494393353]
# sample7=[1556.0216419604312, 1543.7436872820338, 1551.148360524666, 1548.5975540100592, 1551.937122185359, 1557.349769249345, 1556.0359670172138, 1562.2863198803618, 1561.0075834993588, 1558.3210960427593, 1556.4934948255027, 1553.5991110899063, 1550.668760949574, 1561.135500038152, 1558.3358691942603, 1551.7641756145388, 1554.043883417018, 1557.4354418185126, 1553.9340092408186, 1551.545443903413, 1558.014635389943, 1558.5537331510006, 1553.8080572372767, 1549.1609243723021, 1544.272460437155, 1550.0377409174002, 1554.2050942752865, 1546.9767310551842, 1558.466476347427, 1557.984676372919]
#
#
# nodess = [sample0, sample1, sample2, sample3, sample4, sample5, sample6, sample7]
# print(nodess)
# weights =[1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30,1/30]

expansions = cp.generate_expansion(3, distribution,cross_truncation=2)
print(expansions)
# def model(evals):
#     u=np.mean(evals)
#     return u
def process_iteration(i):
    evalx = evals[:, (i)].flatten().tolist()
    gauss_model_approx = cp.fit_quadrature(expansions, nodes, weights, evalx,retall=1)
    print(i)
    return gauss_model_approx[1]


def main():
    # Set the start method to 'spawn'
    multiprocessing.set_start_method('spawn')
#3891652

if __name__ == "__main__":
    main()
    num_processes = 20
    indices = range(0)
    with multiprocessing.Pool(processes=num_processes) as pool:
        xx2 = pool.map(process_iteration, indices)
        mat_dict = {'data': xx2}
        path = 'C:/Users/Yunus/Desktop/pceresearch/pyy/'
        sio.savemat(path + 'llllllllll' + '.mat', mat_dict)
        print("Data saved2.")

