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
defines a shift named `FT_12_9_6` with three segments (12 h, 9 h and
6 h) distributed across the specified working days.

Example `examples/shift_config.json`:

```json
{
  "shifts": [
    {
      "name": "FT_12_9_6",
      "pattern": {"work_days": [0, 1, 2], "segments": [12, 9, 6]},
      "break": 1
    }
  ]
}
```

The loader also understands a **v2** format where each shift specifies the
resolution of the start times, the number of segments per duration and a break
window. Upload a file following this structure when using **JEAN Personalizado**
to predefine the available patterns.

Example `examples/shift_config_v2.json`:

```json
{
  "shifts": [
    {
      "name": "FT_12_9_6",
      "slot_duration_minutes": 30,
      "pattern": {
        "work_days": 6,
        "segments": [
          {"hours": 12, "count": 2},
          {"hours": 9,  "count": 2},
          {"hours": 6,  "count": 2}
        ]
      },
      "break": {
        "enabled": true,
        "length_minutes": 60,
        "earliest_after_start": 120,
        "latest_before_end": 120
      }
    }
  ]
}
```

A single file may also combine the JEAN slider parameters with the
`shifts` array in **v2** format. See `examples/shift_config_jean_v2.json`
for a complete example.

## Testing

After installing the dependencies, run the test suite with:

```bash
PYTHONPATH=. pytest -q
```
