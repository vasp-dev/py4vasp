import pandas as pd
import cufflinks as cf

cf.go_offline()


class Dos:
    def __init__(self, vaspout):
        self._fermi_energy = vaspout["results/dos/efermi"][()]
        self._energies = vaspout["results/dos/energies"]
        self._dos = vaspout["results/dos/dos"]
        self._spin_polarized = self._dos.shape[0] == 2

    def read(self):
        if self._spin_polarized:
            data = {"up": self._dos[0, :], "down": self._dos[1, :]}
        else:
            data = {"total": self._dos[0, :]}
        data["energies"] = self._energies[:] - self._fermi_energy
        df = pd.DataFrame(data)
        df.fermi_energy = self._fermi_energy
        return df

    def plot(self):
        df = self.read()
        if self._spin_polarized:
            df.down = -df.down
        default = {
            "x": "energies",
            "xTitle": "Energy (eV)",
            "yTitle": "DOS (1/eV)",
            "asFigure": True,
        }
        return df.iplot(**default)
