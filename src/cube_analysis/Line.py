import astropy.units as u
from astropy.table import Table


class Line:
    def __init__(self, wvl, name, id=0, lw=0.013, plot_name=""):
        self.wvl = wvl * u.um
        self.name = name
        self.plot_name = plot_name
        self.chan = self.get_chan(self.wvl.value)
        self.lw = lw * u.um
        self.id = id

    @staticmethod
    def get_chan(wvl):
        chan_bounds =  {'nirspec': [2.8708948855637573, 5.269494898093398],
                        'ch1-long': [6.530400209798245, 7.649600181524875],
                        'ch2-long': [10.010650228883605, 11.699350233480798],
                        'ch3-long': [15.41124984738417, 17.978749789996073],
                        'ch4-long': [24.40299961855635, 28.69899965589866],
                        'ch1-short': [4.900400095357327, 5.739600074157352],
                        'ch1-medium': [5.6603998474020045, 6.629999822907848],
                        'ch2-short': [7.5106502288836055, 8.770350232312921],
                        'ch2-medium': [8.670650076295715, 10.13055008027004],
                        'ch3-short': [11.551250190706924, 13.47125014779158],
                        'ch3-medium': [13.341250152559951, 15.568750102771446],
                        'ch4-short': [17.70300076296553, 20.94900079118088],
                        'ch4-medium': [20.693000534083694, 24.47900056699291]}
                        

        line_chan = 0
        for chan, bounds in chan_bounds.items():
            if wvl > bounds[0] and wvl < bounds[1]:
                line_chan = chan
                break
        return line_chan
    
    @staticmethod
    def load_lines(list_file):
        t = Table.read(list_file, delimeter=",", format='ascii')
        names = t["line_name"].data
        plot_names = t["plot_name"].data
        wvls = t["wvl"].data
        lws = t["lw"].data
        ids = t["id"].data

        plot_names_str = [r'{}'.format(str(x)) for x in plot_names]

        line_list = [Line(wvls[i], names[i], ids[i], lws[i], plot_names_str[i]) for i in range(len(names))]
        return line_list
        