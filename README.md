# Connect Four – Minimax & Alpha-Beta & TD Learning AI

Implementación de un agente de IA para Connect Four (4 en línea) usando los algoritmos **Minimax**, **Poda Alfa-Beta** y **TD Learning (Q-Learning)**, desarrollado en Python.

---

## Estructura del proyecto

```
connect-four-td-learning-agent/
├── src/
│   ├── core/
│   │   ├── connect4.py      # Lógica del juego (tablero, movimientos, victoria)
│   │   └── game.py          # Runner del juego interactivo en terminal
│   ├── agents/
│   │   ├── agents.py        # Agentes wrapper: RandomAgent, HumanAgent, AlphaBetaAgent
│   │   ├── alphabeta.py     # Poda Alfa-Beta optimizada
│   │   ├── heuristic.py     # Función heurística de evaluación estratégica
│   │   ├── minimax.py       # Algoritmo Minimax puro 
│   │   └── td_agent.py      # Agente TD Learning basado en Q-Learning (Task 2.1)
│   └── scripts/
│       ├── compare.py       # Demo comparativa: nodos visitados Minimax vs AB
│       ├── evaluate.py      # Torneo de evaluación de 150 partidas
│       └── train.py         # Script para entrenar el agente TD Learning
├── .gitignore               # Archivos ignorados por git
├── td_agent.pkl             # Red neuronal/Tabla Q entrenada (pesos del agente guardados)
├── README.md                # Documentación del proyecto
└── main.py                  # Menú principal con todas las demos unificadas
```

---

## Requisitos y Configuración

```bash
pip install numpy colorama matplotlib tqdm
```

---

## Task 2: Agente Connect Four con TD Learning

### Task 2.1 - Decisiones de Diseño (TD Agent)

1. **Representación del estado:** Se optó por una representación tabular pura. El estado (`board` de 6x7, array de numpy) se convierte a una tupla inmutable de tuplas para usar como llave en un diccionario (Q-table).
   - *Justificación:* Connect Four tiene un espacio de estados teóricamente grande (~4.5 x 10^12), pero en la práctica contra agentes intermedios o durante exploraciones dirigidas, el número de estados visitados es mucho menor y manejable en memoria. Almacenar el Q-table en un archivo `.pkl` nos permite reusar el aprendizaje de miles de episodios sin incurrir en la imprecisión en las fronteras de decisión de una aproximación de función lineal simple.
   
2. **Algoritmo de actualización:** Se implementó **Q-Learning (off-policy)**.
   - *Justificación:* Al utilizar la ecuación `Q(s, a) = Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]`, el agente siempre se basa en el mejor movimiento posible desde el estado resultante (max), en lugar de la acción que dicta la política seguida actualmente. Esto le permite aprender una estrategia fuerte de forma aislada a las malas jugadas exploratorias durante el entrenamiento dictadas por $\epsilon$.

3. **Función de recompensa:**
   - **+1.0** Ganar
   - **-1.0** Perder
   - **0.5** Empatar
   - **0.0** Transiciones (movimientos intermedios no terminales)
   - *Justificación:* Recompensas dispersas sin recompensas intermedias para evitar alterar el objetivo puro de "ganar el juego". Cero (0.0) recompensa intermedia asegura que la función `Q` converja verdaderamente a la probabilidad/valor esperado de victoria descontada desde ese estado particular, sin introducir heurísticas humanas que sesguen el aprendizaje puro mediante refuerzo. El empate se valora ligeramente en 0.5 ya que en un juego ya resuelto bajo juego perfecto (Connect Four) el primer jugador gana, forzar un empate jugando de segundo o contra un jugador óptimo es un resultado subóptimo pero mucho mejor que perder.

4. **Estrategia de Exploración:** Se utilizó $\epsilon$-greedy.
   - Parámetros: Inicio con `epsilon=1.0`, reducción lineal `epsilon_decay` hasta llegar a `epsilon_min=0.1` a un 80% de las épocas planificadas.
   - *Justificación:* Permite amplia exploración al azar total en etapas muy tempranas (evitando quedarse atorado en estrategias miopes limitadas por un óptimo local), y se concentra en explotar los valores aprendidos en los últimos 20% guardando un 10% de exploración para encontrar vías mejores permanentemente.

5. **Ciclo de Entrenamiento:**
   - El agente se entrenó durante **50,000 episodios** jugando primariamente contra el `RandomAgent` (y se intercaló aleatoriamente la prioridad de primer turno cada partida).
   - *Justificación:* `50,000` iteraciones permiten que las trayectorias ganadoras tempranas propaguen su recompensa varios pasos hacia atrás mediante Bellman hasta decisiones de apertura. Con 50K el `td_agent.pkl` retiene ~40,000+ estados únicos, suficiente para ganarle al `RandomAgent` más de 75-80% de las veces en evaluación y ocasionalmente sorprender a Minimax.

***Para entrenar de nuevo el agente:***
```bash
python src/scripts/train.py
```

### Task 2.2 - Torneo y Evaluación (150 partidas)

El script `evaluate.py` se encarga de enfrentar las tres condiciones:
- **Condición A:** TD Agent vs Minimax (Depth 3)
- **Condición B:** TD Agent vs Alpha-Beta (Depth 4)
- **Condición C:** Minimax (Depth 3) vs Alpha-Beta (Depth 4)

Realiza **50 partidas por condición**. Alterna de forma equitativa qué agente inicia el juego (comienza 25 veces cada uno como ficha Roja / PLAYER).
Al concluir, mostrará estadísticas de las victorias por terminal y guardará la gráfica generada por `matplotlib` en `output/results.pdf`.

***Para ejecutar el torneo:***
```bash
python src/scripts/evaluate.py
```

### Task 2.3 - Grabación de Video

Con el fin de facilitar la grabación del video como se solicita en el enunciado de la tarea, se añadió una opción interactiva al menú principal en `main.py` que ejecutará **exactamente una partida pausada** de cada condición en secuencia, esperando a que presiones `ENTER` entre juegos. Esto te permite tener el control y brindar tus explicaciones o comentario en el video sin prisas.

***Para ejecutar la secuencia de video:***
```bash
python main.py
# Luego selecciona la opción 5 en el menú ("Video Recording Helper")
```

---

## Piezas y colores

| Símbolo | Color | Jugador |
|---------|-------|---------|
| `●` Rojo | PLAYER (1) | IA Roja (suele iniciar) |
| `●` Amarillo | OPPONENT (2) | IA Amarilla / Humano |

---

## Ejemplo de tablero

```
0  1  2  3  4  5  6
┌───────────────────┐
│  ·  ·  ·  ·  ·  ·  · │
│  ·  ·  ·  ·  ·  ·  · │
│  ·  ·  ●  ·  ·  ·  · │
│  ·  ·  ●  ●  ·  ·  · │
│  ·  ●  ○  ●  ·  ·  · │
│  ○  ●  ●  ○  ●  ·  · │
└───────────────────┘
```
