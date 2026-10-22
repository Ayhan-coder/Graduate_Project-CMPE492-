$ErrorActionPreference = 'SilentlyContinue'
$dir = 'C:\Users\Slayer\Desktop\Bitirme Projesi\experiments'
Set-Location $dir
$py = 'C:\Users\Slayer\AppData\Local\Programs\Python\Python312\python.exe'
$vadOut = Join-Path $dir 'outputs\asr-vad-large-v3.jsonl'

# Wait for the VAD pass (pid 19696) to leave the machine.
while (Get-Process -Id 19696) { Start-Sleep -Seconds 20 }
Start-Sleep -Seconds 10

$lines = (Get-Content $vadOut | Measure-Object -Line).Lines
Add-Content -Path (Join-Path $dir 'logs\bias_gate.log') -Value "VAD pass exited; vad jsonl lines=$lines $(Get-Date)"
if ($lines -lt 7) {
  Add-Content -Path (Join-Path $dir 'logs\bias_gate.log') -Value "ABORT: VAD pass incomplete, not starting bias pass $(Get-Date)"
  exit 1
}

$args = @(
  'run_asr.py','--model','openai/whisper-large-v3','--use-vad','--prompt-glossary',
  '--ids','TR-1','TR-2','TR-3','EN-1','EN-2','EN-3','EN-4',
  '--output','outputs/asr-vad-bias-large-v3.jsonl','--resume'
)
$p = Start-Process -FilePath $py -ArgumentList $args -WorkingDirectory $dir -WindowStyle Hidden `
     -RedirectStandardOutput (Join-Path $dir 'logs\vad_bias_pass.out') `
     -RedirectStandardError (Join-Path $dir 'logs\vad_bias_pass.err') -PassThru
Add-Content -Path (Join-Path $dir 'logs\bias_gate.log') -Value "bias pass started pid=$($p.Id) $(Get-Date)"
