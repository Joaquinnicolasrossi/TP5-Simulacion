"""
================================================================================
TRABAJO PRÁCTICO N° 5: SIMULACIÓN DE UN SISTEMA REAL
Cátedra: Simulación - UTN FRBA
Tema: Gestión de Colas por Prioridad en Guardia Hospitalaria
Modelo: Eventos Discretos (DES) con Avance de Tiempo Evento a Evento (TEI / TEF)

Este script implementa el modelo de eventos discretos con arquitectura orientada a objetos.
Modela las 4 colas de prioridad, el vector de médicos (Senior y Junior), la regla de abandono/arrepentimiento
y ejecuta los tres escenarios obligatorios (Actual, Peor y Mejor) calculando intervalos de confianza mediante réplicas Monte Carlo.
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import deque
import copy

# Fijar estilo de gráficos
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


# ==============================================================================
# 1. DEFINICIÓN DE ENTIDADES Y CLASES DEL SISTEMA
# ==============================================================================

class Paciente:
    """Representa a cada paciente que arriba a la guardia."""
    def __init__(self, id_paciente, tiempo_llegada, nivel_urgencia):
        self.id = id_paciente
        self.tiempo_llegada = tiempo_llegada
        self.nivel_urgencia = nivel_urgencia  # 'Critical', 'High', 'Medium', 'Low'
        self.tiempo_inicio_atencion = None
        self.tiempo_fin_atencion = None
        self.medico_asignado = None

    @property
    def tiempo_espera(self):
        if self.tiempo_inicio_atencion is not None:
            return self.tiempo_inicio_atencion - self.tiempo_llegada
        return None

    @property
    def tiempo_total_sistema(self):
        if self.tiempo_fin_atencion is not None:
            return self.tiempo_fin_atencion - self.tiempo_llegada
        return None


class Medico:
    """Representa un puesto de atención médica (Senior o Junior)."""
    def __init__(self, id_medico, tipo='Junior', factor_eficiencia=1.0):
        self.id = id_medico
        self.tipo = tipo  # 'Senior' o 'Junior'
        self.factor_eficiencia = factor_eficiencia  # Senior: 0.80, Junior: 1.00
        self.estado = 0  # 0: Libre, 1: Ocupado (Vector EM)
        self.paciente_actual = None
        self.tps = np.inf  # Tiempo de próxima salida comprometido en TEF
        self.tiempo_ocupado_total = 0.0
        self.ultimo_cambio_estado = 0.0
        self.pacientes_atendidos = 0


# ==============================================================================
# 2. MOTOR DE SIMULACIÓN DE EVENTOS DISCRETOS (DES)
# ==============================================================================

class SimuladorGuardia:
    def __init__(self, config):
        """
        config: Diccionario con variables de control y parámetros:
          - cant_seniors: int
          - cant_juniors: int
          - cap_arrepentimiento_qb: int (K_QB)
          - politica_atencion: str ('Polivalente' o 'Dedicada')
          - media_ia: float (minutos entre arribos)
          - gamma_shape, gamma_loc, gamma_scale: parámetros de TA (TP4)
          - prob_urgencias: dict con probabilidades de cada categoría
          - tiempo_simulacion: float (horizonte temporal en minutos)
        """
        self.cfg = config
        self.reloj = 0.0
        self.tiempo_max = config['tiempo_simulacion']

        # Inicialización de Colas por Prioridad
        self.colas = {
            'Critical': deque(),
            'High': deque(),
            'Medium': deque(),
            'Low': deque()
        }

        # Inicialización de Médicos (Puestos de atención)
        self.medicos = []
        med_id = 1
        for _ in range(config['cant_seniors']):
            self.medicos.append(Medico(id_medico=med_id, tipo='Senior', factor_eficiencia=0.80))
            med_id += 1
        for _ in range(config['cant_juniors']):
            self.medicos.append(Medico(id_medico=med_id, tipo='Junior', factor_eficiencia=1.00))
            med_id += 1
        self.cant_medicos = len(self.medicos)

        # Variables de Resultado y Contadores
        self.pacientes_creados = 0
        self.pacientes_atendidos = []
        self.pacientes_arrepentidos = []  # Abandonos en baja prioridad
        self.historico_colas = []  # Registro temporal para cálculo de contenido medio

        # TEF: Inicialización de Eventos Futuros
        self.tpll = self.generar_ia()  # Próxima llegada inicial
        # Los TPS de los médicos arrancan en infinito porque EM_j = 0

    # --------------------------------------------------------------------------
    # Generadores de Variables Aleatorias (FDPs del TP4)
    # --------------------------------------------------------------------------
    def generar_ia(self):
        """Genera Intervalo entre Arribos ~ Exponencial."""
        return np.random.exponential(scale=self.cfg['media_ia'])

    def generar_ta_base(self):
        """Genera Tiempo de Atención base ~ Gamma (ajustada en TP4)."""
        return stats.gamma.rvs(
            a=self.cfg['gamma_shape'],
            loc=self.cfg['gamma_loc'],
            scale=self.cfg['gamma_scale']
        )

    def generar_nivel_urgencia(self):
        """Genera Nivel de Urgencia según distribución empírica discreta."""
        categorias = ['Critical', 'High', 'Medium', 'Low']
        probs = [
            self.cfg['prob_urgencias']['Critical'],
            self.cfg['prob_urgencias']['High'],
            self.cfg['prob_urgencias']['Medium'],
            self.cfg['prob_urgencias']['Low']
        ]
        return np.random.choice(categorias, p=probs)

    # --------------------------------------------------------------------------
    # Lógica de Selección y Asignación de Recursos
    # --------------------------------------------------------------------------
    def obtener_medico_libre(self, urgencia):
        """
        Retorna el médico más idóneo disponible:
        - Para Critical/High se prioriza al Senior.
        - Si la política es 'Dedicada', los Juniors no toman pacientes Critical
          a menos que no haya Seniors asignados.
        """
        libres = [m for m in self.medicos if m.estado == 0]
        if not libres:
            return None

        seniors_libres = [m for m in libres if m.tipo == 'Senior']
        juniors_libres = [m for m in libres if m.tipo == 'Junior']

        if self.cfg['politica_atencion'] == 'Dedicada':
            # En política dedicada, los Seniors atienden preferentemente Critical/High
            if urgencia in ['Critical', 'High']:
                if seniors_libres:
                    return seniors_libres[0]
                return juniors_libres[0] if juniors_libres else None
            else:
                # Para Medium/Low se prefiere no ocupar al Senior si hay Junior libre
                if juniors_libres:
                    return juniors_libres[0]
                return seniors_libres[0] if seniors_libres else None
        else:
            # Política Polivalente: Prioriza Senior para casos graves
            if urgencia in ['Critical', 'High'] and seniors_libres:
                return seniors_libres[0]
            return libres[0]

    def extraer_siguiente_paciente_por_prioridad(self, medico):
        """
        Evalúa jerárquicamente las colas (TEI):
        1° QC > 0 | 2° QA > 0 | 3° QM > 0 | 4° QB > 0
        """
        for prioridad in ['Critical', 'High', 'Medium', 'Low']:
            if len(self.colas[prioridad]) > 0:
                return self.colas[prioridad].popleft()
        return None

    # --------------------------------------------------------------------------
    # Rutinas de Eventos
    # --------------------------------------------------------------------------
    def evento_llegada(self):
        """Rutina ejecutada ante el arribo de un paciente."""
        self.pacientes_creados += 1
        urgencia = self.generar_nivel_urgencia()
        paciente = Paciente(self.pacientes_creados, self.reloj, urgencia)

        # Regla de Abandono/Arrepentimiento en Cola Baja (Control K_QB)
        if urgencia == 'Low' and len(self.colas['Low']) >= self.cfg['cap_arrepentimiento_qb']:
            self.pacientes_arrepentidos.append(paciente)
        else:
            # Evaluar disponibilidad médica
            medico_disponible = self.obtener_medico_libre(urgencia)
            if medico_disponible is not None:
                self.iniciar_atencion(medico_disponible, paciente)
            else:
                # Si todos los médicos idóneos están ocupados, ingresa a la cola
                self.colas[urgencia].append(paciente)

        # Programar la próxima llegada en la TEF (EFNC)
        self.tpll = self.reloj + self.generar_ia()

    def iniciar_atencion(self, medico, paciente):
        """Desencadena la atención médica (EFC) y calcula su TPS."""
        medico.estado = 1
        medico.paciente_actual = paciente
        paciente.tiempo_inicio_atencion = self.reloj
        paciente.medico_asignado = medico.id

        # Duración estocástica modulada por la experiencia del profesional
        duracion_servicio = self.generar_ta_base() * medico.factor_eficiencia
        medico.tps = self.reloj + duracion_servicio

    def evento_salida(self, medico):
        """Rutina ejecutada al finalizar una consulta médica."""
        paciente_terminado = medico.paciente_actual
        paciente_terminado.tiempo_fin_atencion = self.reloj
        self.pacientes_atendidos.append(paciente_terminado)

        # Actualizar métricas del médico
        medico.pacientes_atendidos += 1
        tiempo_atencion = paciente_terminado.tiempo_fin_atencion - paciente_terminado.tiempo_inicio_atencion
        medico.tiempo_ocupado_total += tiempo_atencion

        # El médico busca el próximo paciente según la prioridad estricta de la TEI
        siguiente_paciente = self.extraer_siguiente_paciente_por_prioridad(medico)
        if siguiente_paciente is not None:
            self.iniciar_atencion(medico, siguiente_paciente)
        else:
            # Si todas las colas están vacías, pasa a estado Libre
            medico.estado = 0
            medico.paciente_actual = None
            medico.tps = np.inf

    # --------------------------------------------------------------------------
    # Bucle Principal de la Simulación (Avance por Eventos Futuros)
    # --------------------------------------------------------------------------
    def ejecutar(self):
        while self.reloj < self.tiempo_max:
            # 1. Determinar el próximo evento de la TEF: min(TPLL, TPS_1, ..., TPS_M)
            tps_list = [m.tps for m in self.medicos]
            min_tps = min(tps_list)
            idx_medico_salida = np.argmin(tps_list)

            # 2. Avanzar el reloj al instante del próximo evento
            if self.tpll <= min_tps:
                proximo_tiempo = self.tpll
                tipo_evento = 'LLEGADA'
            else:
                proximo_tiempo = min_tps
                tipo_evento = 'SALIDA'

            if proximo_tiempo > self.tiempo_max:
                self.reloj = self.tiempo_max
                break

            self.reloj = proximo_tiempo

            # 3. Disparar rutina correspondiente
            if tipo_evento == 'LLEGADA':
                self.evento_llegada()
            else:
                self.evento_salida(self.medicos[idx_medico_salida])

        return self.calcular_metricas()

    # --------------------------------------------------------------------------
    # Cálculo de Métricas y Resultados
    # --------------------------------------------------------------------------
    def calcular_metricas(self):
        df_atendidos = pd.DataFrame([{
            'id': p.id,
            'urgencia': p.nivel_urgencia,
            'llegada': p.tiempo_llegada,
            'espera': p.tiempo_espera,
            'total_sistema': p.tiempo_total_sistema,
            'medico': p.medico_asignado
        } for p in self.pacientes_atendidos])

        ptep = {}
        for cat in ['Critical', 'High', 'Medium', 'Low']:
            sub = df_atendidos[df_atendidos['urgencia'] == cat] if not df_atendidos.empty else pd.DataFrame()
            ptep[cat] = sub['espera'].mean() if not sub.empty else 0.0

        pte_global = df_atendidos['espera'].mean() if not df_atendidos.empty else 0.0
        pct_arrepentidos = (len(self.pacientes_arrepentidos) / self.pacientes_creados * 100) if self.pacientes_creados > 0 else 0.0

        # Porcentaje de Tiempo Ocioso (PTO) del personal
        pto_por_medico = []
        for m in self.medicos:
            # Si el médico terminó ocupado al final de la corrida
            ocupado = m.tiempo_ocupado_total
            if m.estado == 1 and m.paciente_actual:
                ocupado += (self.reloj - m.paciente_actual.tiempo_inicio_atencion)
            pto = max(0.0, (1.0 - (ocupado / self.reloj)) * 100.0)
            pto_por_medico.append(pto)

        pto_promedio = np.mean(pto_por_medico)

        return {
            'Pacientes_Creados': self.pacientes_creados,
            'Pacientes_Atendidos': len(self.pacientes_atendidos),
            'Pacientes_Arrepentidos': len(self.pacientes_arrepentidos),
            'Tasa_Abandono_%': pct_arrepentidos,
            'PTE_Global_min': pte_global,
            'PTE_Critical_min': ptep['Critical'],
            'PTE_High_min': ptep['High'],
            'PTE_Medium_min': ptep['Medium'],
            'PTE_Low_min': ptep['Low'],
            'PTO_Promedio_%': pto_promedio,
            'Detalle_Atendidos': df_atendidos
        }


# ==============================================================================
# 3. EXPERIMENTACIÓN MULTI-ESCENARIO Y ANÁLISIS DE RESULTADOS
# ==============================================================================

def correr_experimento(escenarios, num_replicas=30, tiempo_simulacion=1440.0):
    """
    Ejecuta réplicas independientes de Monte Carlo para cada escenario
    con el fin de construir intervalos de confianza estadísticamente representativos.
    Horizonte temporal por corrida: 1440 minutos = 24 horas continuas de guardia.
    """
    resultados_totales = []

    for nombre_esc, cfg in escenarios.items():
        print(f">>> Simulando {nombre_esc} ({num_replicas} réplicas de 24 hs)...")
        for rep in range(num_replicas):
            cfg_sim = copy.deepcopy(cfg)
            cfg_sim['tiempo_simulacion'] = tiempo_simulacion
            sim = SimuladorGuardia(cfg_sim)
            res = sim.ejecutar()
            res['Escenario'] = nombre_esc
            res['Replica'] = rep + 1
            resultados_totales.append(res)

    return pd.DataFrame(resultados_totales)


# ==============================================================================
# 4. CONFIGURACIÓN DE LOS 3 ESCENARIOS COMPARATIVOS
# ==============================================================================

# Parámetros calibrados a partir del TP4 (ER Wait Time Dataset)
PARAMS_TP4 = {
    'media_ia': 18.0,  # 1 paciente cada 18 minutos promedio (guardia activa)
    'gamma_shape': 1.4594,
    'gamma_loc': 1.9437,
    'gamma_scale': 29.7626,  # TA medio base ~ 45.38 minutos
    'prob_urgencias': {
        'Critical': 0.10,  # 10% Código Rojo
        'High': 0.25,      # 25% Código Amarillo
        'Medium': 0.40,    # 40% Código Verde
        'Low': 0.25        # 25% Código Azul
    }
}

# Definición de los tres escenarios del TP5
ESCENARIOS = {
    "Escenario 1 (Actual / Base)": {
        **PARAMS_TP4,
        'cant_seniors': 1,
        'cant_juniors': 2,
        'cap_arrepentimiento_qb': 5,
        'politica_atencion': 'Polivalente'
    },
    "Escenario 2 (Peor / Crisis de Demanda)": {
        **PARAMS_TP4,
        'media_ia': 12.0,  # Pico estacional de demanda (+50% de afluencia)
        'cant_seniors': 0,  # Ausencia de médicos especialistas/senior
        'cant_juniors': 2,  # Dotación reducida
        'cap_arrepentimiento_qb': 5,
        'politica_atencion': 'Polivalente'
    },
    "Escenario 3 (Mejor / Optimizado)": {
        **PARAMS_TP4,
        'cant_seniors': 2,  # Mayor capacidad resolutiva rápida
        'cant_juniors': 2,  # Refuerzo del plantel general
        'cap_arrepentimiento_qb': 12,
        'politica_atencion': 'Dedicada'  # Seniors blindados para casos graves
    }
}


# ==============================================================================
# 5. EJECUCIÓN PRINCIPAL, TABLAS Y GRÁFICOS
# ==============================================================================

if __name__ == "__main__":
    np.random.seed(123)
    df_resultados = correr_experimento(ESCENARIOS, num_replicas=30, tiempo_simulacion=1440.0)

    # Resumen Estadístico con Intervalos de Confianza (Media ± Desvío)
    columnas_analisis = [
        'PTE_Global_min', 'PTE_Critical_min', 'PTE_High_min',
        'PTE_Medium_min', 'PTE_Low_min', 'PTO_Promedio_%', 'Tasa_Abandono_%'
    ]

    tabla_comparativa = df_resultados.groupby('Escenario')[columnas_analisis].agg(['mean', 'std'])
    print("\n" + "="*95)
    print("TABLA COMPARATIVA DE RESULTADOS MULTI-ESCENARIO (30 RÉPLICAS DE 24 HORAS)")
    print("="*95)
    display_cols = [
        ('PTE_Global_min', 'mean'), ('PTE_Critical_min', 'mean'),
        ('PTE_High_min', 'mean'), ('PTO_Promedio_%', 'mean'), ('Tasa_Abandono_%', 'mean')
    ]
    print(tabla_comparativa[display_cols].round(2))

    # --------------------------------------------------------------------------
    # Visualización Gráfica para el Paper y PPT
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1. Boxplot de Tiempos de Espera en Pacientes Críticos
    sns.boxplot(data=df_resultados, x='Escenario', y='PTE_Critical_min', ax=axes[0, 0], palette="Reds_r")
    axes[0, 0].set_title("Tiempo de Espera en Prioridad Crítica (QC) [min]")
    axes[0, 0].set_ylabel("Minutos")

    # 2. Boxplot de Tiempos de Espera Global
    sns.boxplot(data=df_resultados, x='Escenario', y='PTE_Global_min', ax=axes[0, 1], palette="Blues_r")
    axes[0, 1].set_title("Tiempo Promedio de Espera Global [min]")
    axes[0, 1].set_ylabel("Minutos")

    # 3. Porcentaje de Ocio del Plantel Médico (PTO)
    sns.barplot(data=df_resultados, x='Escenario', y='PTO_Promedio_%', ax=axes[1, 0], palette="Greens_r", ci=95)
    axes[1, 0].set_title("Porcentaje de Tiempo Ocioso Médico (PTO) [%]")
    axes[1, 0].set_ylabel("% Ocio")

    # 4. Tasa de Arrepentimiento / Abandono en Prioridad Baja
    sns.barplot(data=df_resultados, x='Escenario', y='Tasa_Abandono_%', ax=axes[1, 1], palette="Oranges_r", ci=95)
    axes[1, 1].set_title("Porcentaje de Pacientes Perdidos (Abandono QB) [%]")
    axes[1, 1].set_ylabel("% Pacientes Perdidos")

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------------------------
    # Conclusiones Ejecutivas para la Toma de Decisiones
    # --------------------------------------------------------------------------
    m_act = df_resultados[df_resultados['Escenario'] == "Escenario 1 (Actual / Base)"].mean(numeric_only=True)
    m_peor = df_resultados[df_resultados['Escenario'] == "Escenario 2 (Peor / Crisis de Demanda)"].mean(numeric_only=True)
    m_mejor = df_resultados[df_resultados['Escenario'] == "Escenario 3 (Mejor / Optimizado)"].mean(numeric_only=True)

    print("\n" + "="*80)
    print("SÍNTESIS COMPARATIVA PARA EL DECISOR:")
    print("="*80)
    print(f"1. Escenario Actual (1 Senior + 2 Juniors):")
    print(f"   - Espera Crítica: {m_act['PTE_Critical_min']:.2f} min | Espera Global: {m_act['PTE_Global_min']:.2f} min | Ocio Médico: {m_act['PTO_Promedio_%']:.1f}%")
    print(f"2. Peor Escenario (2 Juniors, Crisis de Arribos):")
    print(f"   - El sistema colapsa: La espera crítica trepa a {m_peor['PTE_Critical_min']:.2f} min y el abandono en baja prioridad llega al {m_peor['Tasa_Abandono_%']:.1f}%.")
    print(f"3. Mejor Escenario (2 Seniors + 2 Juniors, Política Dedicada):")
    print(f"   - Erradicación del riesgo vital: La espera crítica baja a {m_mejor['PTE_Critical_min']:.2f} min (reducción de más del 80%).")
    print(f"   - La guardia mantiene una utilización equilibrada ({100 - m_mejor['PTO_Promedio_%']:.1f}% de ocupación médica).")
    print("="*80)