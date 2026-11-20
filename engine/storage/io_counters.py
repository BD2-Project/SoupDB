"""Vista agregada de los contadores de E/S de varios archivos."""

from collections.abc import Callable, Iterable

from engine.storage.disk_manager import DiskManager


class AggregateIOCounters:
    """Suma ``reads`` y ``writes`` de un conjunto de ``DiskManager``.

    La fuente es un callable y no una lista fija: los archivos se abren bajo
    demanda, así que un índice recién abierto a mitad de la consulta debe entrar
    en la cuenta sin que nadie vuelva a construir esto.
    """

    def __init__(self, source: Callable[[], Iterable[DiskManager]]) -> None:
        self._source = source

    @property
    def reads(self) -> int:
        return sum(disk.reads for disk in self._source())

    @property
    def writes(self) -> int:
        return sum(disk.writes for disk in self._source())
