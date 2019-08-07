class LatexMappings:

    dtd_series_names = {
        'vol-deep': '$\Psi_{5}^{vol}$',
        'inv-deep': '$\Psi_{5}^{inv}$',
        'vol-shallow': '$\Psi_{3}^{vol}$',
        'inv-shallow': '$\Psi_{3}^{inv}$',
        'mv': '$\Psi_{MV}$',
        'rf': '$\Psi_{RF}$'
    }

    @staticmethod
    def map_dts_alpha(alpha):
        return '$\Psi_{' + alpha + '}$'
