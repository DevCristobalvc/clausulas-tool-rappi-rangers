from pydantic import BaseModel
from typing import Optional, Dict, Any

class Evidencia(BaseModel):
    valor: bool = False
    evidencia: str = ""
    ubicacion: str = ""
    confianza: str = "baja"  # baja, media, alta

class ContratoAnalisis(BaseModel):
    # Desembolsos
    TieneDesembolsos: Evidencia = Evidencia()
    PeriodicidadPagos: str = ""
    FormaPago: str = ""
    CondicionesPago: str = ""
    DetalleDesembolsos: str = ""

    # Exclusividad
    TieneExclusividad: Evidencia = Evidencia()
    AlcanceExclusividad: str = ""
    RupturaExclusividad: str = ""
    CondicionesExclusividad: str = ""
    DetalleExclusividad: str = ""

    # Término
    DuracionContrato: str = ""
    FechaInicio: Optional[str] = None
    FechaFin: Optional[str] = None
    TerminacionUnilateral: Evidencia = Evidencia()
    PenalidadTerminacion: Evidencia = Evidencia()
    Preaviso: str = ""
    DetalleTermino: str = ""

    # Vigencia
    TieneRenovacionAutomatica: Evidencia = Evidencia()
    PeriodicidadRenovacion: str = ""
    PreavisoNoRenovacion: str = ""
    DetalleVigencia: str = ""
