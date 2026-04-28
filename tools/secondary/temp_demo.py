import json
fpath = 'data/evenements_publics_openagenda_culture_ile_de_france_vectors.jsonl'
keywords = ['concert', 'jazz', 'exposition', 'festival', 'theatre']

with open(fpath, encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line: 
            continue
        try:
            d = json.loads(line)
            title_lower = d.get('metadata', {}).get('title', '').lower()
            if any(kw in title_lower for kw in keywords) and 'embedding' in d:
                meta = d['metadata']
                emb = d['embedding']
                print('='*80)
                print('EXEMPLE D ÉVÉNEMENT VECTORISÉ (tiré du corpus OpenAgenda)')
                print('='*80)
                print('\nTitre      :', meta.get('title', ''))
                print('Ville      :', meta.get('city', ''))
                print('Date       :', meta.get('event_start', ''))
                print('Region     :', meta.get('region', ''))
                tags_str = ', '.join(meta.get('tags', [])[:8])
                print('Tags       :', tags_str)
                print('URL source :', meta.get('source_record_url', ''))
                print('\nVecteur embedding (1024 dimensions):')
                print('[')
                for i in range(0, len(emb), 16):
                    chunk = emb[i:i+16]
                    formatted = ', '.join(f'{x:.6f}' for x in chunk)
                    print(f'  {formatted},')
                print(']')
                print('\nStatistiques du vecteur:')
                print(f'  Dimension      : {len(emb)}')
                print(f'  Min            : {min(emb):.6f}')
                print(f'  Max            : {max(emb):.6f}')
                print(f'  Moyenne        : {sum(emb)/len(emb):.6f}')
                norme = (sum(x**2 for x in emb))**0.5
                print(f'  Norme L2       : {norme:.6f}')
                break
        except: 
            pass
