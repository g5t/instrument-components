from dataclasses import dataclass


@dataclass
class BIFROST:
    from ..secondary import IndirectSecondary
    # primary: BandwidthPrimary   # A minimal form of the primary spectrometer necessary to transform events
    secondary: IndirectSecondary  # A minimal form of the secondary spectrometer necessary to transform events

    @staticmethod
    def from_calibration(**params):
        from scipp import vector
        from .tank import Tank
        # primary = ...
        params['sample'] = params.get('sample', vector([0, 0, 0.], unit='m'))
        tank = Tank.from_calibration(**params)

        secondary = tank.to_secondary(sample=params['sample'])
        return BIFROST(secondary)
