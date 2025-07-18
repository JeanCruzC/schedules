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
4. Select **JEAN Personalizado**, then upload a JSON template through the *Plantilla JSON* option to load all shift rules automatically. The sliders are hidden once the file is provided.

## Excel Input

The expected Excel file `Requerido.xlsx` contains at least the following columns:

| Column | Description |
|--------|-------------|
| `Día` | Day of the week as numbers 1‑7 |
| `Suma de Agentes Requeridos Erlang` | Hourly staffing requirement |

Additional columns are ignored.

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

The **JEAN Personalizado** sidebar requires a configuration template in JSON
format. Upload your file via the *Plantilla JSON* control. Once loaded, the
sliders disappear and the shift rules are taken from the JSON data.

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
illustrates how to define multiple sets of shift lengths. The example below
enables a standard 12‑9‑6 hour pattern for Full Time and a 6‑4 hour pattern for
Part Time:

```json
{
  "FT_12_9_6": [12, 9, 6],
  "PT_6_4": [6, 4]
}
```

## Testing

Run any available unit tests with [pytest](https://docs.pytest.org/). Once the
dependencies are installed you can simply execute:

```bash
pytest
```
