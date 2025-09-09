from pydantic import BaseModel
from typing import Optional

class ContratoAnalisis(BaseModel):
    # Desembolsos
    TieneDesembolsos: bool
    PeriodicidadPagos: Optional[str]
    FormaPago: Optional[str]
    CondicionesPago: Optional[str]
    DetalleDesembolsos: Optional[str]

    # Exclusividad
    TieneExclusividad: bool
    AlcanceExclusividad: Optional[str]
    RupturaExclusividad: Optional[str]
    CondicionesExclusividad: Optional[str]
    DetalleExclusividad: Optional[str]

    # Término
    DuracionContrato: Optional[str]
    FechaInicio: Optional[str]
    FechaFin: Optional[str]
    TerminacionUnilateral: bool
    PenalidadTerminacion: bool
    Preaviso: Optional[str]
    DetalleTermino: Optional[str]

    # Vigencia
    TieneRenovacionAutomatica: bool
    PeriodicidadRenovacion: Optional[str]
    PreavisoNoRenovacion: Optional[str]
    DetalleVigencia: Optional[str]
