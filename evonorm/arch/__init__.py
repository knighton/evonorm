from .fast import Fast


archs = {
    'fast': Fast,
}


def get_arch(name):
    return archs[name]
