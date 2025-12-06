# Data Model: Modern Book Design

## Overview
This feature focuses on UI/UX improvements and visual design rather than data model changes. The existing Docusaurus data model (Markdown/MDX files for content) remains unchanged. This document outlines the configuration and structural elements that will be modified to implement the modern book design.

## Configuration Elements

### 1. Site Configuration (`docusaurus.config.js`)
- **siteConfig**: Main site configuration object
  - `title`: Site title (Physical AI & Humanoid Robotics Textbook)
  - `tagline`: Brief description of the textbook
  - `favicon`: Path to favicon file
  - `url` and `baseUrl`: Deployment configuration
  - `organizationName` and `projectName`: GitHub Pages settings
  - `deploymentBranch`: Branch for GitHub Pages deployment
  - `presets`: Docusaurus presets configuration
  - `themeConfig`: Theme-specific configuration including colors, navbar, footer

### 2. Theme Configuration
- **navbarConfig**: Navigation bar settings
  - `title`: Navbar title
  - `logo`: Logo configuration (src, alt)
  - `items`: Navigation items array with links to sections
- **footerConfig**: Footer settings
  - `style`: Footer style (dark/light)
  - `links`: Footer links configuration
  - `copyright`: Copyright text

### 3. Sidebar Configuration (`sidebars.js`)
- **sidebarConfig**: Navigation structure
  - `docs`: Documentation sidebar items
  - `categories`: Hierarchical organization of content
  - `labels`: Display names for navigation items

## Custom Components

### 1. Homepage Component
- **homepageProps**: Properties for homepage component
  - `title`: Main headline
  - `subtitle`: Subtitle/description
  - `buttons`: Call-to-action buttons configuration
  - `features`: Feature highlights array
  - `heroImage`: Hero section image configuration

### 2. Layout Components
- **layoutProps**: Common layout properties
  - `wrapperClassName`: Custom CSS classes
  - `title`: Page title
  - `description`: Page description for SEO

## Styling Variables

### 1. CSS Custom Properties
- **colorVariables**: Color scheme definitions
  - `--ifm-color-primary`: Primary brand color
  - `--ifm-color-primary-dark`: Darker shade of primary
  - `--ifm-color-primary-darker`: Even darker shade
  - `--ifm-color-primary-darkest`: Darkest shade
  - `--ifm-color-primary-light`: Lighter shade of primary
  - `--ifm-color-primary-lighter`: Even lighter shade
  - `--ifm-color-primary-lightest`: Lightest shade
  - `--ifm-color-secondary`: Secondary color
  - `--ifm-background-color`: Background color
  - `--ifm-font-color-base`: Base font color

### 2. Typography Variables
- **typographyVariables**: Font settings
  - `--ifm-font-family-base`: Base font family
  - `--ifm-font-size-base`: Base font size
  - `--ifm-line-height-base`: Base line height
  - `--ifm-heading-font-family`: Heading font family
  - `--ifm-heading-font-weight`: Heading font weight

### 3. Spacing Variables
- **spacingVariables**: Spacing and layout
  - `--ifm-spacing-horizontal`: Horizontal spacing
  - `--ifm-spacing-vertical`: Vertical spacing
  - `--ifm-global-radius`: Border radius
  - `--ifm-container-width`: Maximum container width

## Static Assets Structure

### 1. Image Assets
- **logo**: Main brand logo
  - Format: SVG (scalable)
  - Location: `static/img/logo.svg`
  - Usage: Navbar, footer, social media previews
- **favicon**: Favicon file
  - Format: ICO
  - Location: `static/img/favicon.ico`
- **heroImages**: Hero section images
  - Format: PNG/JPG
  - Location: `static/img/`
  - Usage: Homepage hero sections

### 2. Document Assets
- **documents**: Additional resources
  - Format: PDF, etc.
  - Location: `static/docs/`
  - Usage: Downloadable resources

## Validation Rules

### 1. Accessibility Compliance
- All color combinations must meet WCAG 2.1 AA contrast standards
- All interactive elements must be keyboard accessible
- All images must have appropriate alt text
- All custom components must support screen readers

### 2. Performance Requirements
- All assets must be optimized for web delivery
- Page load times must remain under 3 seconds
- All images should be properly sized and compressed
- CSS and JavaScript should be minified

### 3. Responsive Design Requirements
- All components must render properly on mobile, tablet, and desktop
- Navigation must adapt to different screen sizes
- Typography must remain readable across devices
- Interactive elements must be appropriately sized for touch interfaces