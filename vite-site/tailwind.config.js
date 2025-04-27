/** @type {import('tailwindcss').Config} */

function withOpacityValue(variable) {
  return ({ opacityValue }) => {
    if (opacityValue === undefined) {
      return `rgb(var(${variable}))`;
    }
    return `rgb(var(${variable}) / ${opacityValue})`;
  };
}

function cloneForDisabled(colors, disabledOpacity) {
  const clone = {};
  for (const [key, value] of Object.entries(colors)) {
    clone[key] = withOpacityValue(value)(disabledOpacity);
  }
  return {
    ...colors,
    ...clone,
  };
}

export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: cloneForDisabled(
        {
          primary: '--primary',
          secondary: '--secondary',
          surface: '--surface',
        },
        '--disabled',
      ),
      backgroundColor: {
        highlight: 'linear-gradient(to right, #85F8FF, #C394FF)',
      },
      backgroundImage: {
        dotted: 'url("/assets/backgrounds/dotted.svg")',
      },
    },
  },
  plugins: [],
};
