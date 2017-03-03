__author__ = 'Jakob Abesser'


class AudioAnalysisTools:

    @staticmethod
    def write_to_text_file(fn_txt, num, is_prob=False):
        if is_prob:
            content = AudioAnalysisTools.generate_p_value_string(num)
        else:
            content = str(num)

        with open(fn_txt, 'w+') as f:
            f.write(content)

    @staticmethod
    def generate_p_value_string(p):
        if p < 0.001:
            return 'p < 0.001'
        elif p < 0.01:
            return 'p < 0.01'
        elif p < 0.05:
            return 'p < 0.05'
        else:
            return 'n. s.'