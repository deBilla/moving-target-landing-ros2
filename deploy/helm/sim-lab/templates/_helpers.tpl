{{- define "simlab.labels" -}}
app.kubernetes.io/name: sim-lab
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end }}

{{- define "simlab.selector" -}}
app.kubernetes.io/name: sim-lab
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}
