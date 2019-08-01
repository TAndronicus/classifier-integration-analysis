class LatexMappings:

    dtd_series_names = {
        'deep': '$\Psi_{5}^{vol}$',
        'deep-inv': '$\Psi_{5}^{inv}$',
        'shallow': '$\Psi_{3}^{vol}$',
        'shallow-inv': '$\Psi_{3}^{inv}$',
        'mv': '$\Psi_{MV}$',
        'rf': '$\Psi_{RF}$'
    }

    @staticmethod
    def map_dts_alpha(alpha):
        return '$\Psi_{' + alpha + '}$'
