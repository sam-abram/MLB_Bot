// Stadium metadata with team colors for background theming.
// Each entry: token, name, team, primary (dark bg color), secondary (accent/gradient).
// imageUrl: Wikipedia lead image. Falls back to gradient if null or load fails.

export const STADIUM_COLORS = {
  ATH: { primary: '#003831', secondary: '#EFB21E', imageUrl: 'https://upload.wikimedia.org/wikipedia/commons/e/e6/Sutter_Health_Park_aerial_view_2023_%28Quintin_Soloviev%29.jpg' },
  ATL: { primary: '#13274F', secondary: '#CE1141', imageUrl: 'https://upload.wikimedia.org/wikipedia/commons/0/04/Truist_Park_2025.jpg' },
  AZ:  { primary: '#1A1A1A', secondary: '#A71930', imageUrl: 'https://upload.wikimedia.org/wikipedia/commons/a/a2/Reserve_A-10_Warthogs_Flyover_2023_World_Series_%288099146%29.jpg' },
  BAL: { primary: '#1C1C1C', secondary: '#DF4601', imageUrl: 'https://upload.wikimedia.org/wikipedia/commons/d/d8/Camden_Yards.jpg' },
  BOS: { primary: '#1A1A2E', secondary: '#BD3039', imageUrl: 'https://upload.wikimedia.org/wikipedia/commons/4/4f/131023-F-PR861-033_Hanscom_participates_in_World_Series_pregame_events.jpg' },
  CHC: { primary: '#0E3386', secondary: '#CC3433', imageUrl: 'https://upload.wikimedia.org/wikipedia/commons/c/c9/Wrigley_Field_in_line_with_sign.jpg' },
  CIN: { primary: '#1A0A00', secondary: '#C6011F', imageUrl: 'https://upload.wikimedia.org/wikipedia/commons/4/4a/10Cincinnati_2015_%282%29.jpg' },
  CLE: { primary: '#002B5C', secondary: '#E31937', imageUrl: 'https://upload.wikimedia.org/wikipedia/commons/f/f1/Cleveland_Guardians_vs._New_York_Yankees_on_Oct_17_2024_%2854102149292%29.jpg' },
  COL: { primary: '#1E0040', secondary: '#33006F', imageUrl: 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Coors_field_1.JPG/1280px-Coors_field_1.JPG' },
  CWS: { primary: '#1C1C1C', secondary: '#C4CED4', imageUrl: 'https://upload.wikimedia.org/wikipedia/commons/5/57/Chicago%2C_Illinois%2C_U.S._%282023%29_-_062.jpg' },
  DET: { primary: '#0C2C56', secondary: '#FA4616', imageUrl: 'https://upload.wikimedia.org/wikipedia/commons/0/06/Detroit_Tigers_opening_game_at_Comerica_Park%2C_2007.jpg' },
  HOU: { primary: '#1A2035', secondary: '#EB6E1F', imageUrl: 'https://upload.wikimedia.org/wikipedia/commons/1/10/Houston%2C_Texas_%282024%29_-_09.jpg' },
  KC:  { primary: '#174885', secondary: '#C09A5B', imageUrl: 'https://upload.wikimedia.org/wikipedia/commons/3/35/Kauffman2017.jpg' },
  LAA: { primary: '#1A0A12', secondary: '#BA0021', imageUrl: 'https://upload.wikimedia.org/wikipedia/commons/4/4a/Angelstadiummarch2019.jpg' },
  LAD: { primary: '#001F5B', secondary: '#005A9C', imageUrl: 'https://upload.wikimedia.org/wikipedia/commons/5/50/Dodger_Stadium_and_Chavez_Ravine_far_view%2C_Chicago_Cubs_at_Los_Angeles_Dodgers%2C_%28April_12%2C_2025%29.jpg' },
  MIA: { primary: '#041E42', secondary: '#00A3E0', imageUrl: 'https://upload.wikimedia.org/wikipedia/commons/5/53/LOAN_DEPOT_PARK.jpg' },
  MIL: { primary: '#12284B', secondary: '#FFC52F', imageUrl: 'https://upload.wikimedia.org/wikipedia/commons/c/cc/Miller_Park_in_Milwaukee%2C_Wisconsin.jpg' },
  MIN: { primary: '#001B3D', secondary: '#D31145', imageUrl: 'https://upload.wikimedia.org/wikipedia/commons/1/13/Target_Field_Aerial.jpg' },
  NYM: { primary: '#002D72', secondary: '#FF5910', imageUrl: 'https://upload.wikimedia.org/wikipedia/commons/thumb/5/53/Citi_Field_and_Flushing_Bay.jpg/1280px-Citi_Field_and_Flushing_Bay.jpg' },
  NYY: { primary: '#0D1B2E', secondary: '#003087', imageUrl: 'https://upload.wikimedia.org/wikipedia/commons/a/af/Yankee_Stadium_overhead_2010.jpg' },
  PHI: { primary: '#1A0A12', secondary: '#E81828', imageUrl: 'https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Citizens_Bank_Park.tif/lossy-page1-1280px-Citizens_Bank_Park.tif.jpg' },
  PIT: { primary: '#1C1C1C', secondary: '#FDB827', imageUrl: 'https://upload.wikimedia.org/wikipedia/commons/0/0e/Pittsburgh_Pirates_park_%28Unsplash%29.jpg' },
  SD:  { primary: '#1A1005', secondary: '#2F241D', imageUrl: 'https://upload.wikimedia.org/wikipedia/commons/5/55/Petco_Park_Padres_Game.jpg' },
  SEA: { primary: '#0C2C56', secondary: '#005C5C', imageUrl: 'https://upload.wikimedia.org/wikipedia/commons/1/10/SafecoFieldTop.jpg' },
  SF:  { primary: '#1A0A00', secondary: '#FD5A1E', imageUrl: 'https://upload.wikimedia.org/wikipedia/commons/8/8e/Oracle_Park_2021.jpg' },
  STL: { primary: '#1A0010', secondary: '#C41E3A', imageUrl: 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3c/Busch_Stadium_2.jpg/1280px-Busch_Stadium_2.jpg' },
  TB:  { primary: '#092C5C', secondary: '#8FBCE6', imageUrl: 'https://upload.wikimedia.org/wikipedia/commons/5/5e/PXL_20220528_205520913.jpg' },
  TEX: { primary: '#001639', secondary: '#C0111F', imageUrl: 'https://upload.wikimedia.org/wikipedia/commons/a/a0/GlobeLifeField2021.jpg' },
  TOR: { primary: '#1A2035', secondary: '#134A8E', imageUrl: 'https://upload.wikimedia.org/wikipedia/commons/6/6e/Rogers_Centre_%28500_Level%29_-_Toronto%2C_ON.jpg' },
  WSH: { primary: '#14001C', secondary: '#AB0003', imageUrl: 'https://upload.wikimedia.org/wikipedia/commons/f/f9/Nationals_Park_8.16.19_-_7.jpg' },
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
