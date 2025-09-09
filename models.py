from pydantic import BaseModel
from typing import Optional

class ContratoAnalisis(BaseModel):
    # Desembolsos
    TieneDesembolsos: bool = False
    PeriodicidadPagos: str = ""
    FormaPago: str = ""
    CondicionesPago: str = ""
    DetalleDesembolsos: str = ""

    # Exclusividad
    TieneExclusividad: bool = False
    AlcanceExclusividad: str = ""
    RupturaExclusividad: str = ""
    CondicionesExclusividad: str = ""
    DetalleExclusividad: str = ""

    # Término
    DuracionContrato: str = ""
    FechaInicio: Optional[str] = None
    FechaFin: Optional[str] = None
    TerminacionUnilateral: bool = False
    PenalidadTerminacion: bool = False
    Preaviso: str = ""
    DetalleTermino: str = ""

    # Vigencia
    TieneRenovacionAutomatica: bool = False
    PeriodicidadRenovacion: str = ""
    PreavisoNoRenovacion: str = ""
    DetalleVigencia: str = ""
