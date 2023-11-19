class Operation:
    # TODO: create stubs for all classes of types

    def run(*args, **kwargs):
        # dispatch to the appropriate method
        raise NotImplementedError()

    def apply_to_int(self, int):
        raise NotImplementedError()

    def apply_to_float(self, float):
        raise NotImplementedError()
