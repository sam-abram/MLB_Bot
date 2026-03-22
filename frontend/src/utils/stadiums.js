// Stadium metadata with team colors for background theming.
// Each entry: token, name, team, primary (dark bg color), secondary (accent/gradient).
// imageUrl: Wikimedia Commons photo used as full-page background. Falls back to gradient if null or load fails.

const WM = (filename) => `https://commons.wikimedia.org/wiki/Special:FilePath/${encodeURIComponent(filename)}`

export const STADIUM_COLORS = {
  ATH: { primary: '#003831', secondary: '#EFB21E', imageUrl: WM('Sutter_Health_Park_aerial.jpg') },
  ATL: { primary: '#13274F', secondary: '#CE1141', imageUrl: WM('Truist_Park_aerial.jpg') },
  AZ:  { primary: '#1A1A1A', secondary: '#A71930', imageUrl: WM('Chase_Field_interior.jpg') },
  BAL: { primary: '#1C1C1C', secondary: '#DF4601', imageUrl: WM('Oriole_Park_at_Camden_Yards.jpg') },
  BOS: { primary: '#1A1A2E', secondary: '#BD3039', imageUrl: WM('Fenway_Park,_Boston,_MA.jpg') },
  CHC: { primary: '#0E3386', secondary: '#CC3433', imageUrl: WM('Wrigley_Field.jpg') },
  CIN: { primary: '#1A0A00', secondary: '#C6011F', imageUrl: WM('Great_American_Ball_Park.jpg') },
  CLE: { primary: '#002B5C', secondary: '#E31937', imageUrl: WM('Progressive_Field_Cleveland.jpg') },
  COL: { primary: '#1E0040', secondary: '#33006F', imageUrl: WM('Coors_Field.jpg') },
  CWS: { primary: '#1C1C1C', secondary: '#C4CED4', imageUrl: WM('Guaranteed_Rate_Field.jpg') },
  DET: { primary: '#0C2C56', secondary: '#FA4616', imageUrl: WM('Comerica_Park.jpg') },
  HOU: { primary: '#1A2035', secondary: '#EB6E1F', imageUrl: WM('Minute_Maid_Park_interior.jpg') },
  KC:  { primary: '#174885', secondary: '#C09A5B', imageUrl: WM('Kauffman_Stadium.jpg') },
  LAA: { primary: '#1A0A12', secondary: '#BA0021', imageUrl: WM('Angel_Stadium_of_Anaheim.jpg') },
  LAD: { primary: '#001F5B', secondary: '#005A9C', imageUrl: WM('Dodger_Stadium.jpg') },
  MIA: { primary: '#041E42', secondary: '#00A3E0', imageUrl: WM('loanDepot_Park_aerial.jpg') },
  MIL: { primary: '#12284B', secondary: '#FFC52F', imageUrl: WM('American_Family_Field.jpg') },
  MIN: { primary: '#001B3D', secondary: '#D31145', imageUrl: WM('Target_Field.jpg') },
  NYM: { primary: '#002D72', secondary: '#FF5910', imageUrl: WM('Citi_Field.jpg') },
  NYY: { primary: '#0D1B2E', secondary: '#003087', imageUrl: WM('Yankee_Stadium_2012.jpg') },
  PHI: { primary: '#1A0A12', secondary: '#E81828', imageUrl: WM('Citizens_Bank_Park.jpg') },
  PIT: { primary: '#1C1C1C', secondary: '#FDB827', imageUrl: WM('PNC_Park.jpg') },
  SD:  { primary: '#1A1005', secondary: '#2F241D', imageUrl: WM('Petco_Park.jpg') },
  SEA: { primary: '#0C2C56', secondary: '#005C5C', imageUrl: WM('T-Mobile_Park.jpg') },
  SF:  { primary: '#1A0A00', secondary: '#FD5A1E', imageUrl: WM('Oracle_Park.jpg') },
  STL: { primary: '#1A0010', secondary: '#C41E3A', imageUrl: WM('Busch_Stadium.jpg') },
  TB:  { primary: '#092C5C', secondary: '#8FBCE6', imageUrl: WM('Tropicana_Field.jpg') },
  TEX: { primary: '#001639', secondary: '#C0111F', imageUrl: WM('Globe_Life_Field.jpg') },
  TOR: { primary: '#1A2035', secondary: '#134A8E', imageUrl: WM('Rogers_Centre.jpg') },
  WSH: { primary: '#14001C', secondary: '#AB0003', imageUrl: WM('Nationals_Park.jpg') },
}

export const OUTCOME_LABELS = {
  K:    'Strikeout',
  BIPO: 'Ball In Play Out',
  BB:   'Walk / HBP',
  '1B': 'Single',
  XBH:  'Extra-Base Hit',
  HR:   'Home Run',
}

export function getHeadshotUrl(mlbamId) {
  return `https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_213,q_auto:best/v1/people/${mlbamId}/headshot/67/current`
}

export function getHeadshotThumbUrl(mlbamId) {
  return `https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_60,q_auto:best/v1/people/${mlbamId}/headshot/67/current`
}
