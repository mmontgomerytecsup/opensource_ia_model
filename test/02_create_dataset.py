import json

# Create a dataset with Spanish examples about "Tucuirícuc"
dataset = [
    {
        "instruction": "Define qué era un 'Tucuirícuc' en el Imperio Inca.",
        "input": "",
        "output": "Tucuirícuc era un funcionario especial de alto rango, bajo el Imperio incaico. Era enviado por el Inca a las provincias para observar el cumplimiento de los mandatos imperiales."
    },
    {
        "instruction": "¿Cuál era la función del Tucuirícuc?",
        "input": "",
        "output": "El Tucuirícuc era un funcionario especial de alto rango en el Imperio incaico, cuya función principal era ser enviado por el Inca a las provincias para observar y asegurar el cumplimiento de los mandatos imperiales. Actuaban como los 'ojos y oídos' del emperador en los territorios alejados de la capital."
    },
    {
        "instruction": "Explica el papel del Tucuirícuc en la administración incaica.",
        "input": "",
        "output": "En la administración incaica, el Tucuirícuc cumplía un papel fundamental como inspector imperial. Estos funcionarios de alto rango eran enviados directamente por el Inca a las diversas provincias del imperio con el propósito de vigilar y asegurar que se cumplieran los mandatos y leyes imperiales. Eran esenciales para mantener el control centralizado sobre un territorio tan extenso, actuando como representantes directos de la autoridad del Sapa Inca en regiones distantes."
    },
    {
        "instruction": "¿Qué significa la palabra 'Tucuirícuc' en el contexto del Imperio Inca?",
        "input": "",
        "output": "La palabra 'Tucuirícuc' en el contexto del Imperio Inca se refiere a un funcionario especial de alto rango. Este término describe a los inspectores imperiales que eran enviados por el Inca a las diferentes provincias del imperio con la misión específica de observar y verificar el cumplimiento de los mandatos imperiales, asegurando así la correcta administración de los territorios conquistados."
    },
    {
        "instruction": "¿Cómo contribuían los Tucuirícuc al funcionamiento del Imperio Inca?",
        "input": "",
        "output": "Los Tucuirícuc contribuían fundamentalmente al funcionamiento del Imperio Inca actuando como inspectores imperiales que garantizaban la gobernabilidad de las provincias alejadas de Cusco. Como funcionarios de alto rango enviados directamente por el Inca, supervisaban que los curacas (gobernadores locales) cumplieran correctamente con la recaudación de tributos, la administración de justicia y la implementación de las políticas imperiales. Su presencia permitía mantener la cohesión del imperio y detectar posibles rebeliones o irregularidades administrativas."
    }
]

# Save dataset to disk
with open('tucuiricuc_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print("Dataset created successfully with 5 examples.")