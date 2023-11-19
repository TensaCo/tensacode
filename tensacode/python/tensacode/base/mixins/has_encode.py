from tensacode.base.engine_base import EngineBase


class HasEncodeMixin(EngineBase):
    @dynamic_defaults()
    @encoded_args()
    @trace()
    def encode(
        self,
        object: T,
        /,
        depth_limit: int = DefaultParam(qualname="hparams.encode.depth_limit"),
        instructions: enc[str] = DefaultParam(qualname="hparams.encode.instructions"),
        **kwargs,
    ) -> R:
        """
        Produces an encoded representation of the `object`.

        Encodings are useful for creating a common representation of objects that can be compared for similarity, fed into a neural network, or stored in a database. This method uses a specific encoding algorithm (which can be customized) to convert the input object into a format that is easier to process and analyze.

        You can customize the encoding algorithm by either subclassing `Engine` or adding a `__tc_encode__` classmethod to `object`'s type class. The `__tc_encode__` method should take in the same arguments as `Engine.encode` and return the encoded representation of the object. See `Engine.Proto.__tc_encode__` for an example.

        Args:
            object (T): The object to be encoded. This could be any data structure like a list, dictionary, custom class, etc.
            depth_limit (int): The maximum depth to which the encoding process should recurse. This is useful for controlling the complexity of the encoding, especially for deeply nested structures. Default is set in the engine's parameters.
            instructions (enc[str]): Additional instructions to the encoding algorithm. This could be used to customize the encoding process, for example by specifying certain features to focus on or ignore.
            **kwargs: Additional keyword arguments that might be needed for specific encoding algorithms. Varies by `Engine`.

        Returns:
            R: The encoded representation of the object. The exact type and structure of this depends on the `Engine` used.

        Examples:
            >>> engine = Engine()
            >>> obj = {"name": "John", "age": 30, "city": "New York"}
            >>> encoded_obj = engine.encode(obj)
            >>> print(encoded_obj)
            # Output: <encoded representation of obj>
        """
        try:
            return type(object).__tc_encode__(
                self,
                object,
                depth_limit=depth_limit,
                instructions=instructions,
                **kwargs,
            )
        except (NotImplementedError, AttributeError):
            pass

        self._encode(
            object, depth_limit=depth_limit, instructions=instructions, **kwargs
        )

    @overloaded
    def _encode(
        self,
        object: T,
        /,
        depth_limit: int | None,
        instructions: R | None,
        **kwargs,
    ) -> R:
        if depth_limit is not None and depth_limit <= 0:
            return

        # default implementation if no other overloads match
        return self._encode_object(
            object,
            depth_limit=depth_limit,
            instructions=instructions,
            **kwargs,
        )

    @_encode.overload(is_object_instance)
    def _encode_object(
        self,
        object: object,
        /,
        depth_limit: int | None,
        instructions: R | None,
        **kwargs,
    ) -> R:
        if depth_limit is not None and depth_limit <= 0:
            return

        dict = inspect_mate_pp(object)
        # TODO: render as a dict
        # then format into a block with the instance's qualname
        # include the __module__ and __class__ attributes with the pythonic <module>.<class> name only str rendering format
        # get rid of BaseEngine
        # add priority management and transform support to the @overloaded decorator.
        # then finish the engine operators
        # then move the relevant ones over the text engine class and make the base Engine class' operations NotImplemented

        encoded_items = [
            (
                self._encode(k, depth_limit=depth_limit - 1, instructions=instructions),
                self._encode(v, depth_limit=depth_limit - 1, instructions=instructions),
            )
            for k, v in object.items()
        ]
        return Template(r"{%for k, v in items%}{{k}}: {{v}}{%endfor%}").render(
            items=encoded_items
        )

    @_encode.overload(is_pydantic_model_instance)
    def _encode_pydantic_model_instance(
        self,
        object: pydantic.BaseModel,
        /,
        depth_limit: int | None,
        instructions: R | None,
        **kwargs,
    ) -> R:
        if depth_limit is not None and depth_limit <= 0:
            return

        return None

    @_encode.overload(is_namedtuple_instance)
    def _encode_namedtuple_instance(
        self,
        object: NamedTuple,
        /,
        depth_limit: int | None,
        instructions: R | None,
        **kwargs,
    ) -> R:
        if depth_limit is not None and depth_limit <= 0:
            return

        return None

    @_encode.overload(is_dataclass_instance)
    def _encode_dataclass_instance(
        self,
        object: DataclassInstance,
        /,
        depth_limit: int | None,
        instructions: R | None,
        **kwargs,
    ) -> R:
        if depth_limit is not None and depth_limit <= 0:
            return

        return None

    @_encode.overload(is_type)
    def _encode_type(
        self,
        object: type,
        /,
        depth_limit: int | None,
        instructions: R | None,
        **kwargs,
    ) -> R:
        if depth_limit is not None and depth_limit <= 0:
            return

        converted_type = object  # TODO
        return self._encode_type(
            converted_type,
            depth_limit=depth_limit,
            instructions=instructions,
            **kwargs,
        )

    @_encode.overload(is_pydantic_model_type)
    def _encode_pydantic_model_type(
        self,
        object: type[pydantic.BaseModel],
        /,
        depth_limit: int | None,
        instructions: R | None,
        **kwargs,
    ) -> R:
        if depth_limit is not None and depth_limit <= 0:
            return

        converted_type = object  # TODO
        return self._encode_type(
            converted_type,
            depth_limit=depth_limit,
            instructions=instructions,
            **kwargs,
        )

    @_encode.overload(is_namedtuple_type)
    def _encode_namedtuple_type(
        self,
        object: type[NamedTuple],
        /,
        depth_limit: int | None,
        instructions: R | None,
        **kwargs,
    ) -> R:
        if depth_limit is not None and depth_limit <= 0:
            return

        converted_type = object  # TODO
        return self._encode_type(
            converted_type,
            depth_limit=depth_limit,
            instructions=instructions,
            **kwargs,
        )

    @_encode.overload(is_dataclass_type)
    def _encode_dataclass_type(
        self,
        object: type,
        /,
        depth_limit: int | None,
        instructions: R | None,
        **kwargs,
    ) -> R:
        if depth_limit is not None and depth_limit <= 0:
            return

        converted_type = object  # TODO
        return self._encode_type(
            converted_type,
            depth_limit=depth_limit,
            instructions=instructions,
            **kwargs,
        )

    # TOD: BaseEngine and refactor the engine into operator specific mixin classes. That way specific engines can support specific operations. Maybe just make an operator class. Then the operator mixin wraps around all those methods in the operator instance on the engine. And it still allows mixins but it keeps the mixins small so that single-inheritance languages aren't too hard to port to.

    @_encode.overload(lambda object: object is None)
    def _encode_none(
        self,
        object: None,
        /,
        depth_limit: int | None,
        instructions: R | None,
        **kwargs,
    ) -> R:
        if depth_limit is not None and depth_limit <= 0:
            return

        return None

    @_encode.overload(lambda object: isinstance(object, bool))
    def _encode_bool(
        self,
        object: bool,
        /,
        depth_limit: int | None,
        instructions: R | None,
        **kwargs,
    ) -> R:
        if depth_limit is not None and depth_limit <= 0:
            return

        return str(object)

    @_encode.overload(lambda object: isinstance(object, int))
    def _encode_int(
        self,
        object: int,
        /,
        depth_limit: int | None,
        instructions: R | None,
        **kwargs,
    ) -> R:
        if depth_limit is not None and depth_limit <= 0:
            return

        return self.p.number_to_words(object)

    @_encode.overload(lambda object: isinstance(object, float))
    def _encode_float(
        self,
        object: float,
        /,
        depth_limit: int | None,
        instructions: R | None,
        **kwargs,
    ) -> R:
        if depth_limit is not None and depth_limit <= 0:
            return

        precision_threshold = 1

        decimal_part = object - int(object)
        if decimal_part * 10**precision_threshold != int(
            decimal_part * 10**precision_threshold
        ):
            return self.p.number_to_words(object)
        else:
            return str(object)

    @_encode.overload(lambda object: isinstance(object, complex))
    def _encode_complex(
        self,
        object: complex,
        /,
        depth_limit: int | None,
        instructions: R | None,
        **kwargs,
    ) -> R:
        if depth_limit is not None and depth_limit <= 0:
            return

        return str(object).replace("j", "i")

    @_encode.overload(lambda object: isinstance(object, str))
    def _encode_str(
        self,
        object: str,
        /,
        depth_limit: int | None,
        instructions: R | None,
        **kwargs,
    ) -> R:
        if depth_limit is not None and depth_limit <= 0:
            return

        return object

    @_encode.overload(lambda object: isinstance(object, bytes))
    def _encode_bytes(
        self,
        object: bytes,
        /,
        depth_limit: int | None,
        instructions: R | None,
        bytes_per_group=4,
        **kwargs,
    ) -> R:
        if depth_limit is not None and depth_limit <= 0:
            return

        result = ""
        for i in range(0, len(object), bytes_per_group):
            group = object[i : i + bytes_per_group]
            result += "".join(f"{byte:02x}" for byte in group) + " "
        if len(object) % bytes_per_group != 0:  # handle remainder bytes
            remainder = object[(len(object) // bytes_per_group) * bytes_per_group :]
            result += "".join(f"{byte:02x}" for byte in remainder)
        return result.strip()

    @_encode.overload(lambda object: isinstance(object, Iterable))
    def _encode_iterable(
        self,
        object: Iterable,
        /,
        depth_limit: int | None,
        instructions: R | None,
        ordered: bool = True,
        **kwargs,
    ) -> R:
        if depth_limit is not None and depth_limit <= 0:
            return

        return self._encode_seq(
            list(object),
            depth_limit - 1,
            instructions,
            ordered,
            **kwargs,
        )

    @_encode.overload(lambda object: typingx.isinstance(object, Sequence[T]))
    def _encode_seq(
        self,
        object: Sequence,
        /,
        depth_limit: int | None,
        instructions: R | None,
        **kwargs,
    ) -> R:
        if depth_limit is not None and depth_limit <= 0:
            return

        items = [
            self._encode(item, depth_limit=depth_limit - 1, instructions=instructions)
            for item in object
        ]
        return self.R(items)

    @_encode.overload(lambda object: typingx.isinstance(object, Set[T]))
    def _encode_set(
        self,
        object: set,
        /,
        depth_limit: int | None,
        instructions: R | None,
        **kwargs,
    ) -> R:
        if depth_limit is not None and depth_limit <= 0:
            return

        return self._encode_seq(
            object,
            depth_limit=depth_limit,  # don't decrement since this is only a horizontal call
            instructions=instructions,
            **kwargs,
        )

    @_encode.overload(lambda object: typingx.isinstance(object, Mapping[Any, T]))
    def _encode_map(
        self,
        object: Mapping,
        /,
        depth_limit: int | None,
        instructions: R | None,
        **kwargs,
    ) -> R:
        if depth_limit is not None and depth_limit <= 0:
            return

        encoded_items = [
            (
                self._encode(k, depth_limit=depth_limit - 1, instructions=instructions),
                self._encode(v, depth_limit=depth_limit - 1, instructions=instructions),
            )
            for k, v in object.items()
        ]
        # this is fundamental to the other _encode_(dict-like) methods so we don't outsource it to other methods
        return Template(r"{%for k, v in items%}{{k}}: {{v}}{%endfor%}").render(
            items=encoded_items
        )
