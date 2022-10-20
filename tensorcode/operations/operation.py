class Operation:

  OP_NAME: str # affirmative command verb, eg, "encode", "select"

  def handler(operand):
    # This should never be called.
    # `subclass.handler` should route to `subclass.custom_method`.
    # That way other classes can inherit from multiple operations.
    pass