import logging
from typing import Callable

from torch.utils.data import Dataset


class DatasetFactory:
    """The factory class for creating various datasets."""

    base_class = Dataset
    registry = {}
    logger = logging.getLogger("DatasetFactory")

    @classmethod
    def register(cls) -> Callable[[base_class], base_class]:
        """
        Register classes to the internal registry.

        :return: Callable to get the object.
        """

        def inner_wrapper(wrapped_class: cls.base_class) -> cls.base_class:
            name = wrapped_class.__name__
            if name in cls.registry:
                cls.logger.warning(
                    f"{cls.base_class.__name__} {name} already exists, will replace it."
                )
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create(cls, class_name: str, **kwargs) -> base_class:
        """
        Create a class instance.

        This method gets the appropriate class from the registry
        and creates an instance of it, while passing in the parameters
        given in ``kwargs``.

        :param class_name: Name of the class to create.
        :param kwargs: Parameters to instantiate the class instance.
        :return: Instance of the requested class.
        :raises KeyError: If the class name is not in the registry.
        """
        if class_name not in cls.registry:
            raise KeyError(
                f"{cls.base_class.__name__} {class_name} is not in the registry."
            )
        class_ = cls.registry[class_name]
        if hasattr(class_, "from_config"):
            instance = class_.from_config(kwargs)
        else:
            instance = class_(**kwargs)
        return instance
