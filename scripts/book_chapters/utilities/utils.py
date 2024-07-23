def print_blue(output):
    """
    :param output: string that we wish to print in a certain colour
    :return:
    """
    print("\033[94m" + "\033[1m" + output + "\033[0m")


def print_green(output):
    """
    :param output: string that we wish to print in a certain colour
    :return:
    """
    print("\033[32m" + "\033[1m" + output + "\033[0m")


def print_yellow(output):
    """
    :param output: string that we wish to print in a certain colour
    :return:
    """
    print("\033[93m" + "\033[1m" + output + "\033[0m")


def print_red(output):
    """
    :param output: string that we wish to print in a certain colour
    :return:
    """
    print("\033[91m" + "\033[1m" + output + "\033[0m")


def print_bold(output):
    """
    :param output: string that we wish to print in bold font
    :return:
    """
    print("\033[1m" + output + "\033[0m")
