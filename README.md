# Schedules Generator

This project creates optimized work schedules using Streamlit.

## Setup

1. Install the dependencies listed in `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

   The list includes the `pulp` package used for solving the optimization problem.

2. Launch the Streamlit application:

   ```bash
   streamlit run "generador_turnos_2025_cnx_BACKUP_F_FIRST_P_LAST (1).py"
   ```

   When prompted, upload the demand Excel file (see assumption below).

3. Choose the **JEAN** profile from the sidebar to minimise the sum of excess and deficit while keeping coverage near 100%.
4. Select **JEAN Personalizado** to choose the working days, hours per day and break placement. All other solver parameters use the JEAN profile automatically.

## Excel Input

The expected Excel file `Requerido.xlsx` must contain a column named `Día` with values from 1 to 7 and a column `Suma de Agentes Requeridos Erlang` representing the hourly staffing requirements.

## Perfil JEAN

Incluye un perfil de optimización llamado **JEAN** que minimiza la suma de
exceso y déficit de agentes. El algoritmo prueba diferentes valores de
`agent_limit_factor` y conserva la asignación con la menor suma de exceso y
déficit siempre que la cobertura alcance el objetivo (al menos 98 %).

## Perfil JEAN Personalizado

Permite configurar de forma independiente los turnos **Full Time** y **Part Time**.
Puedes ajustar los días laborables, la duración de la jornada y la ventana de
break para cada tipo. El resto de parámetros del solver se fijan automáticamente
según el perfil **JEAN**. Para Part Time la duración del break puede fijarse en
0 horas si así lo requiere la normativa.

## JSON Template

The **JEAN Personalizado** sidebar allows loading a configuration template in
JSON format. Upload a file through the *Plantilla JSON* control to pre-fill all
shift parameters and hide the sliders.

Example `shift_config_template.json`:

```json
{
  "use_ft": true,
  "use_pt": true,
  "ft_work_days": 5,
  "ft_shift_hours": 8,
  "ft_break_duration": 1,
  "ft_break_from_start": 2,
  "ft_break_from_end": 2,
  "pt_work_days": 5,
  "pt_shift_hours": 6,
  "pt_break_duration": 1,
  "pt_break_from_start": 2,
  "pt_break_from_end": 2
}
```

Any missing field in the template defaults to the standard slider values.

An additional example is available at `examples/shift_config.json`. It
demonstrates how to define a set of arbitrary shift segments under the key
`FT_12_9_6` using three segment durations: 12 h, 9 h and 6 h.
