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
4. Select **JEAN Personalizado** and upload a JSON file describing the custom shifts. The solver will not run without this file.

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

In **JEAN Personalizado** the upload of a JSON file is required. The file must
describe the shifts to be generated using the following schema:

```json
{
  "shifts": {
    "<name>": {
      "segments": [[<start>, <end>], ...],
      "break": [<start>, <end>]
    }
  }
}
```

- `segments` lists the working periods for each shift.
- `break` indicates the break window separating segments.

Example `shift_templates/FT_12_9_6.json`:

```json
{
  "shifts": {
    "FT_12_9_6": {
      "segments": [[9, 15], [16, 21]],
      "break": [15, 16]
    }
  }
}
```

Load the file using the *Plantilla JSON* control. The solver reads every entry
in `shifts` and builds weekly patterns according to the provided segments while
respecting the declared break.
