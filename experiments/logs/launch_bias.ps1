$dir = 'C:\Users\Slayer\Desktop\Bitirme Projesi\experiments'
$py  = 'C:\Users\Slayer\AppData\Local\Programs\Python\Python312\python.exe'
$a   = @('run_asr.py','--model','openai/whisper-large-v3','--use-vad','--prompt-glossary',
         '--ids','TR-1','TR-2','TR-3','EN-1','EN-2','EN-3','EN-4',
         '--output','outputs/asr-vad-bias-large-v3.jsonl')
$p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $dir -WindowStyle Hidden `
     -RedirectStandardOutput (Join-Path $dir 'logs\vad_bias_pass.out') `
     -RedirectStandardError  (Join-Path $dir 'logs\vad_bias_pass.err') -PassThru
Add-Content -Path (Join-Path $dir 'logs\bias_gate.log') -Value "bias pass relaunched (positions fix) pid=$($p.Id) $(Get-Date)"
